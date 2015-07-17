// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>


#ifndef STATS_HPP
#define STATS_HPP

namespace isa {
namespace utils {

template< typename T > class Stats {
public:
	Stats();
	~Stats();

	inline void addElement(T element);
	inline void reset();

	inline long long unsigned int getNrElements() const;
	inline double getAverage() const;
	inline double getVariance() const;
	inline double getStdDev() const;

private:
	long long unsigned int nrElements;
	double average;
	double variance;
};

// Implementations

template< typename T > Stats< T >::Stats() : nrElements(0), average(0.0), variance(0.0) {}

template< typename T > Stats< T >::~Stats() {}

template< typename T > inline void Stats< T >::addElement(T element) {
	double oldAverage = average;

	nrElements++;

	if ( nrElements == 1 ) {
		average = static_cast< double >(element);
		return;
	}
	average = oldAverage + ((element - oldAverage) / nrElements);
	variance += (element - oldAverage) * (element - average);
}

template< typename T > inline void Stats< T >::reset() {
	nrElements = 0;
	average = 0.0;
	variance = 0.0;
}

template< typename T > inline long long unsigned int Stats< T >::getNrElements() const {
	return nrElements;
}

template< typename T > inline double Stats< T >::getAverage() const {
	return average;
}

template< typename T > inline double Stats< T >::getVariance() const {
	return variance;
}

template< typename T > inline double Stats< T >::getStdDev() const {
	return sqrt(variance / nrElements);
}

} // utils
} // isa

#endif
