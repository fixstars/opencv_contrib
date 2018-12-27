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

#ifndef SGM_WINNER_TAKES_ALL_HPP
#define SGM_WINNER_TAKES_ALL_HPP

#include "opencv2/core/cuda.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace stereosgm
    {
        template <size_t MAX_DISPARITY>
        void winnerTakesAll(const GpuMat& src, GpuMat& left, GpuMat& right, float uniqueness, bool subpixel, cv::cuda::Stream& stream);
    }
}}}

#endif
