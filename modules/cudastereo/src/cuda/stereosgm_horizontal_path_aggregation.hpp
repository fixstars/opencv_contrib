/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef SGM_HORIZONTAL_PATH_AGGREGATION_HPP
#define SGM_HORIZONTAL_PATH_AGGREGATION_HPP

#include "opencv2/core/cuda.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace stereosgm
    {

template <unsigned int MAX_DISPARITY>
void aggregateLeft2RightPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	cv::cuda::Stream stream);

template <unsigned int MAX_DISPARITY>
void aggregateRight2LeftPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	cv::cuda::Stream stream);

}
}}}

#endif
