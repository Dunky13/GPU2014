 # define __CL_ENABLE_EXCEPTIONS
 # include < CL/cl.h >
 # include < iostream >
 # include < string >
 # include < iomanip >
 # include < CImg.h >
 # include < Timer.hpp >

// Sequential kernel
void kernel_h(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned char * outputImage);
// OpenCL kernel
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

void kernel_d(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned char * outputImage) {
	LOFAR::NSTimer kernelTimer("Kernel", false, false);
	LOFAR::NSTimer memoryTimer("Memory", false, false);
	LOFAR::NSTimer globalTimer("Global", false, false);

	unsigned int specspace = width * height;
	unsigned int N = spectrum * specspace;
	cl::Buffer * devA = 0;
	cl::Buffer * devC = 0;
	cl::Event clEvent;
	cl::Kernel * kernel = 0;
	vector < cl::Platform >  * platforms = new vector < cl::Platform > ();
	cl::Context * context = new cl::Context();
	vector < cl::Device >  * devices = new vector < cl::Device > ();
	vector < cl::CommandQueue >  * queues = new vector < cl::CommandQueue > ();
	NSTimer globalTimer("GlobalTimer", false, false);
	NSTimer kernelTimer("KernelTimer", false, false);
	NSTimer memoryTimer("MemoryTimer", false, false);

	// Initialize OpenCL
	try {
		unsigned int nrDevices = 0;
		cl::Platform::get(platforms);
		cl_context_properties properties[] = {
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)(platforms->at(clPlatformID))(),
			0
		};
		 * context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

		 * devices = context->getInfo < CL_CONTEXT_DEVICES > ();
		nrDevices = devices->size();
		for (unsigned int device = 0; device < nrDevices; device++) {
			queues->push_back(cl::CommandQueue( * context, devices->at(device)));
		}
	} catch (cl::Error & err) {
		cerr << "Impossible to initialize OpenCL." << endl;
		return 1;
	}

	// Start of the computation
	globalTimer.start();

	// Allocate OpenCL memory
	try {
		devA = new cl::Buffer( * context, CL_MEM_READ_ONLY, N * sizeof(unsigned char), NULL, NULL);
		devC = new cl::Buffer( * context, CL_MEM_READ_WRITE, N * sizeof(unsigned char), NULL, NULL);
	} catch (cl::Error & err) {
		cerr << "Impossible to allocate device memory." << endl;
		return 1;
	}

	// Copy input to device
	memoryTimer.start();
	try {
		(queues->at(clDeviceID)).enqueueWriteBuffer( * devA, CL_TRUE, 0, N * sizeof(unsigned char), reinterpret_cast < void *  > (inputImage), NULL,  & clEvent);
		clEvent.wait();
	} catch (cl::Error & err) {
		cerr << "Impossible to copy memory to device." << endl;
		return 1;
	}
	memoryTimer.stop();

	// Compile the kernel
	string code = 	"__kernel void transpose(__global unsigned char * inputImage, unsigned char * outputImage, const unsigned int N, const unsigned int specspace, const unsigned int width, const unsigned int height) {\n"\n"
					"	int index = get_global_id(0);\n"
					"	if (index < N) {\n"
					"		int index2 = index % specspace;\n"
					"		int outIndex = (index2 % width) * height + ((int)((float)index2 / width)) + ((int)((float)i / specspace)) * specspace;\n"
					"		outputImage[outIndex] = inputImage[index];\n"
					"	}\n"
					"}\n";
	try {
		cl::Program * program = 0;
		cl::Program::Sources sources(1, make_pair(code.c_str(), code.length()));
		program = new cl::Program( * context, sources, NULL);
		program->build(vector < cl::Device > (1, devices->at(clDeviceID)), "-cl-mad-enable", NULL, NULL);
		kernel = new cl::Kernel( * program, "transpose", NULL);
		delete program;
	} catch (cl::Error & err) {
		cerr << "Impossible to build and create the kernel." << endl;
		return 1;
	}

	// Execute the kernel
	cl::NDRange globalSize(static_cast < unsigned int > (ceil(N / static_cast < float > (nrThreads))) * nrThreads);
	cl::NDRange localSize(nrThreads);
	kernel->setArg(0,  * devA);
	kernel->setArg(1,  * devC);
	kernel->setArg(2, N);
	kernel->setArg(3, specspace);
	kernel->setArg(4, width);
	kernel->setArg(5, height);

	kernelTimer.start();
	try {
		(queues->at(clDeviceID)).enqueueNDRangeKernel( * kernel, cl::NullRange, globalSize, localSize, NULL,  & clEvent);
		clEvent.wait();
	} catch (cl::Error & err) {
		cerr << "Impossible to run the kernel" << endl;
		return 1;
	}
	kernelTimer.stop();

	// Copy the output back to host
	memoryTimer.start();
	try {
		(queues->at(clDeviceID)).enqueueReadBuffer( * devC, CL_TRUE, 0, N * sizeof(unsigned char), reinterpret_cast < void *  > (outputImage), NULL,  & clEvent);
		clEvent.wait();
	} catch (cl::Error & err) {
		cerr << "Impossible to copy the results back to host." << endl;
		return 1;
	}
	memoryTimer.stop();

	// End of the computation
	globalTimer.stop();

	// Print performance metrics
	std::cout << "OpenCL" << std::endl;
	std::cout << "Time (g): \t" << std::fixed << std::setprecision(6) << globalTimer.getElapsed() << std::endl;
	std::cout << "Time (m): \t" << std::fixed << std::setprecision(6) << memoryTimer.getElapsed() << std::endl;
	std::cout << "Time (k): \t" << std::fixed << std::setprecision(6) << kernelTimer.getElapsed() << std::endl;
	std::cout << "GFLOP/s: \t" << std::fixed << std::setprecision(3) << "-" << std::endl;
	std::cout << "GB/s: \t\t" << std::fixed << std::setprecision(3) << "-" << std::endl;
}

__kernel void transpose(__global unsigned char *inputImage, __global unsigned char *outputImage, const unsigned int N, const unsigned int specspace, const unsigned int width, const unsigned int height) {
	int index = get_global_id(0);
	if (index < N) {
		int index2 = index % specspace;
		int outIndex = (index2 % width) * height + ((int)((float)index2 / width)) + ((int)((float)index / specspace)) * specspace;
		outputImage[outIndex] = inputImage[index];
	}
}