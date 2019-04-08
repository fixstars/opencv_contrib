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

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/calib3d.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace stereosgm
    {
        namespace {
            template<typename SRC_T, typename DST_T>
            __global__ void check_consistency_kernel(PtrStep<DST_T> d_leftDisp, const PtrStep<DST_T> d_rightDisp, const PtrStep<SRC_T> d_left, int width, int height, bool subpixel)  {

                const int j = blockIdx.x * blockDim.x + threadIdx.x;
                const int i = blockIdx.y * blockDim.y + threadIdx.y;

                // left-right consistency check, only on leftDisp, but could be done for rightDisp too

                SRC_T mask = d_left(i, j);
                int d = d_leftDisp(i, j);
                if (subpixel) {
                    d >>= StereoMatcher::DISP_SHIFT;
                }
                int k = j - d;
                if (mask == 0 || d <= 0 || (k >= 0 && k < width && abs(d_rightDisp(i, k) - d) > 1)) {
                    // masked or left-right inconsistent pixel -> invalid
                    d_leftDisp(i, j) = 0;
                }
            }

            template <typename disp_type, typename image_type>
            void check_consistency(PtrStep<disp_type> d_left_disp, const PtrStep<disp_type> d_right_disp, const PtrStep<image_type> d_src_left, int width, int height, bool subpixel, Stream& _stream)
            {
                const dim3 blocks(width / 16, height / 16);
                const dim3 threads(16, 16);
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);

                check_consistency_kernel<image_type, disp_type><<<blocks, threads>>>(d_left_disp, d_right_disp, d_src_left, width, height, subpixel);

                cudaSafeCall( cudaGetLastError() );
            }
        }

        void checkConsistency(GpuMat& left_disp, const GpuMat& right_disp, const GpuMat& src_left, bool subpixel, Stream& stream)
        {
            Size size = left_disp.size();
            CV_Assert(size == right_disp.size());
            CV_Assert(size == src_left.size());
            CV_Assert(left_disp.type() == CV_16UC1);
            CV_Assert(left_disp.type() == right_disp.type());
            CV_Assert(src_left.type() == CV_8UC1 || src_left.type() == CV_16UC1);

            switch (src_left.type())
            {
            case CV_8UC1:
                check_consistency<uint16_t, uint8_t>(left_disp, right_disp, src_left, size.width, size.height, subpixel, stream);
                break;
            case CV_16UC1:
                check_consistency<uint16_t, uint16_t>(left_disp, right_disp, src_left, size.width, size.height, subpixel, stream);
                break;
            default:
                CV_Error(cv::Error::BadDepth, "Unsupported depth");
            }
        }

    }
}}}
