#include <iostream>
#include <string>
#include <iomanip>
#include <CImg.h>
#include <Timer.hpp>
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using LOFAR::NSTimer;

const unsigned int nrThreads = 256;
// Sequential kernel
void kernel_h(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned char * outputImage);
// CUDA kernel
void kernel_d(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned char * outputImage);

int main(int argc, char * argv[]) {
	if (argc != 2) {
		std::cerr << argv[0] << " <input_image>" << std::endl;
		return 1;
	}

	// Load input image
	cimg_library::CImg < unsigned char > inputImage = cimg_library::CImg < unsigned char > (argv[1]);
	// Prepare output
	cimg_library::CImg < unsigned char > outputImage_h = cimg_library::CImg < unsigned char > (inputImage.height(), inputImage.width(), 1, inputImage.spectrum());
	cimg_library::CImg < unsigned char > outputImage_d = cimg_library::CImg < unsigned char > (inputImage.height(), inputImage.width(), 1, inputImage.spectrum());

	// Lauch kernels
	kernel_h(inputImage.width(), inputImage.height(), inputImage.spectrum(), inputImage.data(), outputImage_h.data());
	kernel_d(inputImage.width(), inputImage.height(), inputImage.spectrum(), inputImage.data(), outputImage_d.data());

	// Correctness check
	long long unsigned int wrongItems = 0;

	for (int x = 0; x < outputImage_h.width(); x++) {
		for (int y = 0; y < outputImage_h.height(); y++) {
			for (int c = 0; c < outputImage_h.spectrum(); c++) {
				if (outputImage_h(x, y, 0, c) != outputImage_d(x, y, 0, c)) {
					wrongItems++;
				}
			}
		}
	}
	if (wrongItems > 0) {
		std::cout << "Wrong: \t\t" << wrongItems << std::fixed << std::setprecision(2) << " (" << (wrongItems * 100.0) / (inputImage.width() * inputImage.height() * inputImage.spectrum()) << "%)" << std::endl;
	}

	return 0;
}

void kernel_h(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned char * outputImage) {
	LOFAR::NSTimer kernelTimer("Kernel", false, false);

	kernelTimer.start();
	// Kernel
	for (unsigned int c = 0; c < spectrum; c++) {
		for (unsigned int y = 0; y < height; y++) {
			for (unsigned int x = 0; x < width; x++) {
				outputImage[(c * width * height) + (x * height) + y] = inputImage[(c * width * height) + (y * width) + x];
			}
		}
	}
	// /Kernel
	kernelTimer.stop();

	// Print performance metrics
	std::cout << "Sequential" << std::endl;
	std::cout << "Time: \t\t" << std::fixed << std::setprecision(6) << kernelTimer.getElapsed() << std::endl;
	std::cout << "GFLOP/s: \t" << std::fixed << std::setprecision(3) << "-" << std::endl;
	std::cout << "GB/s: \t\t" << std::fixed << std::setprecision(3) << (2 * sizeof(unsigned char) * static_cast < long long unsigned int > (width) * height * spectrum) / 1000000000.0 / kernelTimer.getElapsed() << std::endl;
}

__global__ void transpose(const unsigned int N, const unsigned int spectrum, const unsigned int width, const unsigned int height, unsigned char *inputImage, unsigned char *outputImage) {
	unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned int origIndex = index;
	if (index >= N) return;
	
	
	unsigned int 	c 		= (index - (index % (width * height))) / (width * height);
						index 	= (index - (c * (width * height)));
	unsigned int 	y 		= (index - (index % (width))) / (width);
	unsigned int 	x	 	= (index - (y * width));

	outputImage[(c * (width * height)) + (x * height) + y] = inputImage[origIndex];
	
}

void kernel_d(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned char * outputImage) {
	LOFAR::NSTimer kernelTimer("Kernel", false, false);
	LOFAR::NSTimer memoryTimer("Memory", false, false);
	LOFAR::NSTimer globalTimer("Global", false, false);

	cudaError_t devRetVal = cudaSuccess;
	unsigned char * devA = 0;
	unsigned char * devC = 0;
	
	unsigned long int specspace = width * height;
	unsigned int N = spectrum * specspace;
	
	// Start of the computation
	globalTimer.start();

	// Allocate CUDA memory
	if ((devRetVal = cudaMalloc(reinterpret_cast < void **  > ( & devA), N * sizeof(unsigned char))) != cudaSuccess) {
		cerr << "Impossible to allocate device memory for inputImage." << endl;
		return;
	}

	if ((devRetVal = cudaMalloc(reinterpret_cast < void **  > ( & devC), N * sizeof(unsigned char))) != cudaSuccess) {
		cerr << "Impossible to allocate device memory for outputImage." << endl;
		return;
	}
	// Copy input to device
	memoryTimer.start();
	if ((devRetVal = cudaMemcpy(devA, (inputImage), N * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess) {
		cerr << "Impossible to copy devA to device." << endl;
		return;
	}

	memoryTimer.stop();

	// Execute the kernel
	// dim3 blockSize(nrThreads/8, 8);
	// dim3 gridSize(ceil((spectrum * width) / blockSize.x), ceil(height / blockSize.y));
	// int factor = nrThreads;
	// dim3 blockSize(ceil(width/(double)factor), factor/64, 64);
	// dim3 gridSize(spectrum, ceil(height/(double)factor), factor);
	
	dim3 blockSize(nrThreads, spectrum);
	dim3 gridSize(ceil(width/(double)nrThreads), height);	
	
	printf("Block x %d, y %d, z%d\n Grid x %d, y %d, z%d\n", blockSize.x, blockSize.y, blockSize.z, gridSize.x,gridSize.y,gridSize.z);
	
	kernelTimer.start();
	transpose <<<gridSize,blockSize>>> (N, spectrum, width, height, devA, devC);
	cudaDeviceSynchronize();
	kernelTimer.stop();

	// Check if the kernel returned an error
	if ((devRetVal = cudaGetLastError()) != cudaSuccess) {
		cerr << "Uh, the kernel had some kind of issue: " << cudaGetErrorString(devRetVal) << endl;
		return;
	}

	// Copy the output back to host
	memoryTimer.start();
	if ((devRetVal = cudaMemcpy(reinterpret_cast < void *  > (outputImage), devC, N * sizeof(unsigned char), cudaMemcpyDeviceToHost)) != cudaSuccess) {
		cerr << "Impossible to copy devC to host." << cudaGetErrorString(devRetVal) << endl;
		return;
	}
	memoryTimer.stop();

	// End of the computation
	globalTimer.stop();

	// Print performance metrics
	std::cout << "CUDA" << std::endl;
	std::cout << "Time (g): \t" << std::fixed << std::setprecision(6) << globalTimer.getElapsed() << std::endl;
	std::cout << "Time (m): \t" << std::fixed << std::setprecision(6) << memoryTimer.getElapsed() << std::endl;
	std::cout << "Time (k): \t" << std::fixed << std::setprecision(6) << kernelTimer.getElapsed() << std::endl;
	std::cout << "GFLOP/s: \t" << std::fixed << std::setprecision(3) << "-" << std::endl;
	std::cout << "GB/s: \t\t" << std::fixed << std::setprecision(3) << "-" << std::endl;

	cudaFree(devA);
	cudaFree(devC);

	return;
}