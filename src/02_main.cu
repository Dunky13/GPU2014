
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <Timer.hpp>
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using LOFAR::NSTimer;
// const unsigned int CHANNELS = 256;
// const unsigned int SAMPLES = 1024;
// const unsigned int STATIONS = 128;
// const unsigned int BEAMS = 512;
const unsigned int DIV = 1;

const unsigned int CHANNELS = 256/DIV;
const unsigned int SAMPLES = 1024/DIV;
const unsigned int STATIONS = 128/DIV;
const unsigned int BEAMS = 512/DIV;

// Sequential kernel
void kernel_h(const float * weights, const float * input, float * output);
// CUDA kernel
void kernel_d(const float * weights, const float * input, float * output);
// Floating point comparison
inline bool same(const float a, const float b);

int main(int argc, char * argv[]) {
  // Generate input
  std::vector< float > weights(CHANNELS * STATIONS * BEAMS);
  std::vector< float > input(STATIONS * CHANNELS * SAMPLES);
  std::vector< float > output_h(BEAMS * CHANNELS * SAMPLES);
  std::vector< float > output_d(BEAMS * CHANNELS * SAMPLES);

  std::srand(std::time(NULL));
  for ( unsigned int channel = 0; channel < CHANNELS; channel++ ) {
    for ( unsigned int station = 0; station < STATIONS; station++ ) {
      for ( unsigned int beam = 0; beam < BEAMS; beam++ ) {
        weights[(channel * STATIONS * BEAMS) + (station * BEAMS) + beam] = static_cast< float >(std::rand() % 10);
      }
      for ( unsigned int sample = 0; sample < SAMPLES; sample++ ) {
        input[(station * CHANNELS * SAMPLES) + (channel * SAMPLES) + sample] = static_cast< float >(std::rand() % 100);
      }
    }
  }
  std::fill(output_h.begin(), output_h.end(), 0.0f);
  std::fill(output_d.begin(), output_d.end(), 0.0f);


  // Lauch kernels
  kernel_h(weights.data(), input.data(), output_h.data());
  kernel_d(weights.data(), input.data(), output_d.data());
  
  // Correctness check
  long long unsigned int wrongItems = 0;

  for ( unsigned int beam = 0; beam < BEAMS; beam++ ) {
    for ( unsigned int channel = 0; channel < CHANNELS; channel++ ) {
      for ( unsigned int sample = 0; sample < SAMPLES; sample++ ) {
        if ( !same(output_h[(beam * CHANNELS * SAMPLES) + (channel * SAMPLES) + sample], output_d[(beam * CHANNELS * SAMPLES) + (channel * SAMPLES) + sample]) ) {
          wrongItems++;
        }
      }
    }
  }
  
  if ( wrongItems > 0 ) {
    std::cout << "Wrong: \t\t" << wrongItems << std::fixed << std::setprecision(2) << " (" << (wrongItems * 100.0) / (static_cast< long long unsigned int >(BEAMS) * CHANNELS * SAMPLES) << "%)" << std::endl;
  }
  else{
	std::cout << "Nothing Wrong." << std::endl;
  }

  return 0;
}

void kernel_h(const float * weights, const float * input, float * output) {
  LOFAR::NSTimer kernelTimer("Kernel", false, false);
  kernelTimer.start();
  // Kernel
  for ( unsigned int beam = 0; beam < BEAMS; beam++ ) {
    for ( unsigned int channel = 0; channel < CHANNELS; channel++ ) {
      for ( unsigned int sample = 0; sample < SAMPLES; sample++ ) {
        for ( unsigned int station = 0; station < STATIONS; station++ ) {
          output[(beam * CHANNELS * SAMPLES) + (channel * SAMPLES) + sample] += input[(station * CHANNELS * SAMPLES) + (channel * SAMPLES) + sample] * weights[(channel * STATIONS * BEAMS) + (station * BEAMS) + beam];
        }
        output[(beam * CHANNELS * SAMPLES) + (channel * SAMPLES) + sample] /= STATIONS;
      }
    }
  }
  // /Kernel
  kernelTimer.stop();

  // Print performance metrics
  std::cout << "Sequential" << std::endl;
  std::cout << "Time: \t\t" << std::fixed << std::setprecision(6) << kernelTimer.getElapsed() << std::endl;
  std::cout << "GFLOP/s: \t" << std::fixed << std::setprecision(3) << ((static_cast< long long unsigned int >(BEAMS) * CHANNELS * SAMPLES * STATIONS * 2) + (static_cast< long long unsigned int >(BEAMS) * CHANNELS * SAMPLES)) / 1000000000.0 / kernelTimer.getElapsed() << std::endl;
  std::cout << "GB/s: \t\t" << std::fixed << std::setprecision(3) << ((static_cast< long long unsigned int >(BEAMS) * CHANNELS * SAMPLES * STATIONS * 3 * sizeof(float)) + (static_cast< long long unsigned int >(BEAMS) * CHANNELS * SAMPLES * 2 * sizeof(float))) / 1000000000.0 / kernelTimer.getElapsed() << std::endl;
}

__global__ void radio(float *input, float *weights, float *output, const unsigned int BEAMS, const unsigned int CHANNELS, const unsigned int SAMPLES, const unsigned int STATIONS) {
	unsigned long long int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	unsigned long long int index = blockId * blockDim.x + threadIdx.x;
	
	unsigned int 			N1 		= STATIONS;
	unsigned int 			N12 	= SAMPLES 	* N1;
	unsigned long int 		N123 	= CHANNELS 	* N12;
	unsigned long long int 	N1234 	= BEAMS 	* N123;
	
	if(index >= N1234) return;
		
	unsigned int 			beamIndex 		= (index - (index % (N123)))/(N123);
							index			= (index - (beamIndex * N123));
	unsigned int 			channelIndex 	= (index - (index % (N12))) / (N12);
							index 			= (index - (channelIndex * N12));
	unsigned int 			sampleIndex 	= (index - (index % (N1))) / (N1);
	unsigned int 			stationIndex 	= (index - (sampleIndex * N1));
	
	atomicAdd(&output[(beamIndex * CHANNELS * SAMPLES) + (channelIndex * SAMPLES) + sampleIndex], 
				input[(stationIndex * CHANNELS * SAMPLES) + (channelIndex * SAMPLES) + sampleIndex] * 
				weights[(channelIndex * STATIONS * BEAMS) + (stationIndex * BEAMS) + beamIndex]);
	__syncthreads();
	if(stationIndex == 0){ //Whole block writes to one output channel - to prevent multiple divisions, only the first of the block may divide.
		output[(beamIndex * CHANNELS * SAMPLES) + (channelIndex * SAMPLES) + sampleIndex] /= STATIONS;
	}
}

void kernel_d(const float * weights, const float * input, float * output) {
	LOFAR::NSTimer kernelTimer("Kernel", false, false);
	LOFAR::NSTimer memoryTimer("Memory", false, false);
	LOFAR::NSTimer globalTimer("Global", false, false);

	cudaError_t devRetVal = cudaSuccess;
	float * devA = 0;
	float * devB = 0;
	float * devC = 0;
		
	// Start of the computation
	globalTimer.start();

	// Allocate CUDA memory
	if ((devRetVal = cudaMalloc(reinterpret_cast < void **  > ( & devA), (STATIONS * CHANNELS * SAMPLES) * sizeof(float))) != cudaSuccess) {
		cerr << "Impossible to allocate device memory for inputImage." << endl;
		return;
	}
	if ((devRetVal = cudaMalloc(reinterpret_cast < void **  > ( & devB), (CHANNELS * STATIONS * BEAMS) * sizeof(float))) != cudaSuccess) {
		cerr << "Impossible to allocate device memory for inputImage." << endl;
		return;
	}

	if ((devRetVal = cudaMalloc(reinterpret_cast < void **  > ( & devC), (BEAMS * CHANNELS * SAMPLES) * sizeof(float))) != cudaSuccess) {
		cerr << "Impossible to allocate device memory for outputImage." << endl;
		return;
	}
	// Copy input to device
	memoryTimer.start();
	if ((devRetVal = cudaMemcpy(devA, (input), (STATIONS * CHANNELS * SAMPLES) * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
		cerr << "Impossible to copy devA to device." << endl;
		return;
	}
	if ((devRetVal = cudaMemcpy(devB, (weights), (CHANNELS * STATIONS * BEAMS) * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
		cerr << "Impossible to copy devA to device." << endl;
		return;
	}
	if ((devRetVal = cudaMemset(devC, 0, (BEAMS * CHANNELS * SAMPLES) * sizeof(float))) != cudaSuccess) {
		cerr << "Impossible to zero devC." << endl;
		return;
	}
	memoryTimer.stop();

	// Execute the kernel
	dim3 blockSize(STATIONS);
	dim3 gridSize(BEAMS*CHANNELS,SAMPLES);//num of loops
	kernelTimer.start();
	radio <<< gridSize, blockSize>>> (devA, devB, devC, BEAMS, CHANNELS, SAMPLES, STATIONS);
	cudaDeviceSynchronize();
	kernelTimer.stop();

	// Check if the kernel returned an error
	if ((devRetVal = cudaGetLastError()) != cudaSuccess) {
		cerr << "Uh, the kernel had some kind of issue: " << cudaGetErrorString(devRetVal) << endl;
		return;
	}

	// Copy the output back to host
	memoryTimer.start();
	if ((devRetVal = cudaMemcpy(reinterpret_cast < void *  > (output), devC, (BEAMS * CHANNELS * SAMPLES) * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
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
	std::cout << "GFLOP/s: \t" << std::fixed << std::setprecision(3) << ((static_cast< long long unsigned int >(BEAMS) * CHANNELS * SAMPLES * STATIONS * 2) + (static_cast< long long unsigned int >(BEAMS) * CHANNELS * SAMPLES)) / 1000000000.0 / kernelTimer.getElapsed() << std::endl;
  std::cout << "GB/s: \t\t" << std::fixed << std::setprecision(3) << ((static_cast< long long unsigned int >(BEAMS) * CHANNELS * SAMPLES * STATIONS * 3 * sizeof(float)) + (static_cast< long long unsigned int >(BEAMS) * CHANNELS * SAMPLES * 2 * sizeof(float))) / 1000000000.0 / memoryTimer.getElapsed() << std::endl;

	cudaFree(devA);
	cudaFree(devC);
}

inline bool same(const float a, const float b) {
  return abs(a - b) < 1e-6;
}

