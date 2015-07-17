
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>
#include <CImg.h>
#include <Timer.hpp>

const unsigned int SHADES = 256;

// Sequential kernel
void kernel_h(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned int * histograms);
// OpenCL kernel
void kernel_d(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned int * histograms);


int main(int argc, char * argv[]) {
  if ( argc != 2 ) {
    std::cerr << argv[0] << " <input_image>" << std::endl;
    return 1;
  }

  // Load input image
  cimg_library::CImg< unsigned char > inputImage = cimg_library::CImg< unsigned char >(argv[1]);
  std::vector< unsigned int > histograms_h(inputImage.spectrum() * SHADES);
  std::vector< unsigned int > histograms_d(inputImage.spectrum() * SHADES);

  std::fill(histograms_h.begin(), histograms_h.end(), 0);
  std::fill(histograms_d.begin(), histograms_d.end(), 0);

  // Lauch kernels
  kernel_h(inputImage.width(), inputImage.height(), inputImage.spectrum(), inputImage.data(), histograms_h.data());
  kernel_d(inputImage.width(), inputImage.height(), inputImage.spectrum(), inputImage.data(), histograms_d.data());
  
  // Correctness check
  long long unsigned int wrongItems = 0;

  for ( int c = 0; c < inputImage.spectrum(); c++ ) {
    for ( unsigned int i = 0; i < SHADES; i++ ) {
      if ( histograms_h[(c * SHADES) + i] != histograms_d[(c * SHADES) + i] ) {
        wrongItems++;
      }
    }
  }
  
  if ( wrongItems > 0 ) {
    std::cout << "Wrong: \t\t" << wrongItems << std::fixed << std::setprecision(2) << " (" << (wrongItems * 100.0) / (inputImage.spectrum() *  SHADES) << "%)" << std::endl;
  }

  return 0;
}

void kernel_h(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned int * histograms) {
  LOFAR::NSTimer kernelTimer("Kernel", false, false);

  kernelTimer.start();
  // Kernel
  for ( unsigned int c = 0; c < spectrum; c++ ) {
    for ( unsigned int y = 0; y < height; y++ ) {
      for ( unsigned int x = 0; x < width; x++ ) {
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
std::cout << "GB/s: \t\t" << std::fixed << std::setprecision(3) << ((sizeof(unsigned char) * static_cast< long long unsigned int >(width) * height * spectrum) + (2 * sizeof(unsigned int) * static_cast< long long unsigned int >(width) * height * spectrum)) / 1000000000.0 / kernelTimer.getElapsed() << std::endl;
}

void kernel_d(const unsigned int width, const unsigned int height, const unsigned int spectrum, const unsigned char * inputImage, unsigned int * histograms) {
  LOFAR::NSTimer kernelTimer("Kernel", false, false);
  LOFAR::NSTimer memoryTimer("Memory", false, false);
  LOFAR::NSTimer globalTimer("Global", false, false);

  kernelTimer.start();
  // Kernel
  for ( unsigned int c = 0; c < spectrum; c++ ) {
    for ( unsigned int y = 0; y < height; y++ ) {
      for ( unsigned int x = 0; x < width; x++ ) {
        histograms[(c * SHADES) + inputImage[(c * width * height) + (y * width) + x]] += 1;
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

