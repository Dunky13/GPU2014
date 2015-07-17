
#include <iostream>
#include <vector>
#include <algorithm>
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
const unsigned int SHADES = 256;
const unsigned int nrThreads = 512;

// Sequential kernel
void kernel_h(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned int * histograms);
// CUDA kernel
void kernel_d(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned int * histograms);

int main(int argc, char * argv[]) {
	if (argc != 2) {
		std::cerr << argv[0] << " <input_image>" << std::endl;
		return 1;
	}

	// Load input image
	cimg_library::CImg < unsigned char > inputImage = cimg_library::CImg < unsigned char > (argv[1]);
	std::vector < unsigned int > histograms_h(inputImage.spectrum() * SHADES);
	std::vector < unsigned int > histograms_d(inputImage.spectrum() * SHADES);

	std::fill(histograms_h.begin(), histograms_h.end(), 0);
	std::fill(histograms_d.begin(), histograms_d.end(), 0);

	// Lauch kernels
	kernel_h(inputImage.width(), inputImage.height(), inputImage.spectrum(), inputImage.data(), histograms_h.data());
	kernel_d(inputImage.width(), inputImage.height(), inputImage.spectrum(), inputImage.data(), histograms_d.data());

	// Correctness check
	long long unsigned int wrongItems = 0;

	for (int c = 0; c < inputImage.spectrum(); c++) {
		for (unsigned int i = 0; i < SHADES; i++) {
			if (histograms_h[(c * SHADES) + i] != histograms_d[(c * SHADES) + i]) {
				//printf("Index: %d, Sequential: %d, CUDA: %d\n", (c * SHADES) + i, histograms_h[(c * SHADES) + i],histograms_d[(c * SHADES) + i]);
				wrongItems++;
			}
		}
	}

	if (wrongItems > 0) {
		std::cout << "Wrong: \t\t" << wrongItems << std::fixed << std::setprecision(2) << " (" << (wrongItems * 100.0) / (inputImage.spectrum() * SHADES) << "%)" << std::endl;
	}

	return 0;
}

void kernel_h(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned int * histograms) {
	LOFAR::NSTimer kernelTimer("Kernel", false, false);

	kernelTimer.start();
	// Kernel
	for (unsigned int c = 0; c < spectrum; c++) {
		for (unsigned int y = 0; y < height; y++) {
			for (unsigned int x = 0; x < width; x++) {
				histograms[(c * SHADES) + inputImage[(c * width * height) + (y * width) + x]] += 1;
			}
		}
	}
	// /Kernel
	kernelTimer.stop();

	// Print performance metrics
	std::cout << "Sequential" << std::endl;
	std::cout << "Time: \t\t" << std::fixed << std::setprecision(6) << kernelTimer.getElapsed() << std::endl;
	std::cout << "GFLOP/s: \t" << std::fixed << std::setprecision(3) << "-" << std::endl;
	std::cout << "GB/s: \t\t" << std::fixed << std::setprecision(3) << ((sizeof(unsigned char) * static_cast < long long unsigned int > (width) * height * spectrum) + (2 * sizeof(unsigned int) * static_cast < long long unsigned int > (width) * height * spectrum)) / 1000000000.0 / kernelTimer.getElapsed() << std::endl;
}

__global__ void transpose(const unsigned int N, const unsigned int specspace, const unsigned int shades, unsigned char * inputImage, unsigned int * histograms) {
	int index = blockIdx.x * blockDim.x + threadIdx.x ;
	
	if (index < N) {
		unsigned int c = (int)((double)index / specspace);
		atomicAdd(&histograms[(c * shades) + inputImage[index]], 1);
	}
}



void kernel_d(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned int * histograms) {
	LOFAR::NSTimer kernelTimer("Kernel", false, false);
	LOFAR::NSTimer memoryTimer("Memory", false, false);
	LOFAR::NSTimer globalTimer("Global", false, false);

	cudaError_t devRetVal = cudaSuccess;
	unsigned char * devA = 0;
	unsigned int * devC = 0;
	
	unsigned int specspace = width * height;
	unsigned int N = spectrum * specspace;
	
	cudaDeviceProp prop;
	if ((devRetVal =  cudaGetDeviceProperties( &prop, 0 ) )!= cudaSuccess) {
		cerr << "Error getting blocks" << endl;
		return;
	}
	int blocks = prop.multiProcessorCount;
	
	// Start of the computation
	globalTimer.start();

	// Allocate CUDA memory
	if ((devRetVal = cudaMalloc(reinterpret_cast < void **  > ( & devA), N * sizeof(unsigned char))) != cudaSuccess) {
		cerr << "Impossible to allocate device memory for inputImage." << endl;
		return;
	}

	if ((devRetVal = cudaMalloc(reinterpret_cast < void **  > ( & devC), (spectrum * SHADES) * sizeof(unsigned int))) != cudaSuccess) {
		cerr << "Impossible to allocate device memory for outputImage." << endl;
		return;
	}
	// Copy input to device
	memoryTimer.start();
	if ((devRetVal = cudaMemcpy(devA, (inputImage), N * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess) {
		cerr << "Impossible to copy devA to device." << endl;
		return;
	}
	if ((devRetVal = cudaMemset(devC, 0, (spectrum * SHADES) * sizeof(unsigned int))) != cudaSuccess) {
		cerr << "Impossible to zero devC." << endl;
		return;
	}
	memoryTimer.stop();

	// Execute the kernel
	dim3 gridSize(static_cast < unsigned int > (ceil(N / static_cast < float > (nrThreads))));
	dim3 blockSize(SHADES*spectrum);

	kernelTimer.start();
	transpose <<< gridSize, blockSize>>> (N, specspace, SHADES, devA, devC);
	cudaDeviceSynchronize();
	kernelTimer.stop();

	// Check if the kernel returned an error
	if ((devRetVal = cudaGetLastError()) != cudaSuccess) {
		cerr << "Uh, the kernel had some kind of issue: " << cudaGetErrorString(devRetVal) << endl;
		return;
	}

	// Copy the output back to host
	memoryTimer.start();
	if ((devRetVal = cudaMemcpy(reinterpret_cast < void *  > (histograms), devC, (spectrum * SHADES) * sizeof(unsigned int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
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
}

