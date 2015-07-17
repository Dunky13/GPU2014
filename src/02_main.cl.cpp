
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <Timer.hpp>

const unsigned int CHANNELS = 256;
const unsigned int SAMPLES = 1024;
const unsigned int STATIONS = 128;
const unsigned int BEAMS = 512;

// Sequential kernel
void kernel_h(const float * weights, const float * input, float * output);
// OpenCL kernel
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

void kernel_d(const float * weights, const float * input, float * output) {
  LOFAR::NSTimer kernelTimer("Kernel", false, false);
  LOFAR::NSTimer memoryTimer("Memory", false, false);
  LOFAR::NSTimer globalTimer("Global", false, false);

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
  std::cout << "OpenCL" << std::endl;
  std::cout << "Time (g): \t" << std::fixed << std::setprecision(6) << globalTimer.getElapsed() << std::endl;
  std::cout << "Time (m): \t" << std::fixed << std::setprecision(6) << memoryTimer.getElapsed() << std::endl;
  std::cout << "Time (k): \t" << std::fixed << std::setprecision(6) << kernelTimer.getElapsed() << std::endl;
  std::cout << "GFLOP/s: \t" << std::fixed << std::setprecision(3) << "-" << std::endl;
  std::cout << "GB/s: \t\t" << std::fixed << std::setprecision(3) << "-" << std::endl;
}

inline bool same(const float a, const float b) {
  return abs(a - b) < 1e-6;
}

