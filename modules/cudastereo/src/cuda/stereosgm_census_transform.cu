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

#if !defined CUDA_DISABLER

#include "stereosgm_census_transform.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include <cstdint>

namespace cv { namespace cuda { namespace device
{
    namespace stereosgm
    {
        namespace {
        static constexpr int WINDOW_WIDTH  = 9;
        static constexpr int WINDOW_HEIGHT = 7;

        static constexpr int BLOCK_SIZE = 128;
        static constexpr int LINES_PER_BLOCK = 16;

        template <typename T>
        __global__ void census_transform_kernel(
            PtrStepSz<T> src,
            PtrStep<int32_t> dest)
        {
            using pixel_type = T;
            static const int SMEM_BUFFER_SIZE = WINDOW_HEIGHT + 1;

            const int half_kw = WINDOW_WIDTH  / 2;
            const int half_kh = WINDOW_HEIGHT / 2;

            __shared__ pixel_type smem_lines[SMEM_BUFFER_SIZE][BLOCK_SIZE];

            const int tid = threadIdx.x;
            const int x0 = blockIdx.x * (BLOCK_SIZE - WINDOW_WIDTH + 1) - half_kw;
            const int y0 = blockIdx.y * LINES_PER_BLOCK;

            for(int i = 0; i < WINDOW_HEIGHT; ++i){
                const int x = x0 + tid, y = y0 - half_kh + i;
                pixel_type value = 0;
                if(0 <= x && x < src.cols && 0 <= y && y < src.rows){
                    value = src(y, x);
                }
                smem_lines[i][tid] = value;
            }
            __syncthreads();

#pragma unroll
            for(int i = 0; i < LINES_PER_BLOCK; ++i){
                if(i + 1 < LINES_PER_BLOCK){
                    // Load to smem
                    const int x = x0 + tid, y = y0 + half_kh + i + 1;
                    pixel_type value = 0;
                    if(0 <= x && x < src.cols && 0 <= y && y < src.rows){
                        value = src(y, x);
                    }
                    const int smem_x = tid;
                    const int smem_y = (WINDOW_HEIGHT + i) % SMEM_BUFFER_SIZE;
                    smem_lines[smem_y][smem_x] = value;
                }

                if(half_kw <= tid && tid < BLOCK_SIZE - half_kw){
                    // Compute and store
                    const int x = x0 + tid, y = y0 + i;
                    if(half_kw <= x && x < src.cols - half_kw && half_kh <= y && y < src.rows - half_kh){
                        const int smem_x = tid;
                        const int smem_y = (half_kh + i) % SMEM_BUFFER_SIZE;
                        uint32_t f = 0;
                        for(int dy = -half_kh; dy < 0; ++dy){
                            const int smem_y1 = (smem_y + dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
                            const int smem_y2 = (smem_y - dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
                            for(int dx = -half_kw; dx <= half_kw; ++dx){
                                const int smem_x1 = smem_x + dx;
                                const int smem_x2 = smem_x - dx;
                                const auto a = smem_lines[smem_y1][smem_x1];
                                const auto b = smem_lines[smem_y2][smem_x2];
                                f = (f << 1) | (a > b);
                            }
                        }
                        for(int dx = -half_kw; dx < 0; ++dx){
                            const int smem_x1 = smem_x + dx;
                            const int smem_x2 = smem_x - dx;
                            const auto a = smem_lines[smem_y][smem_x1];
                            const auto b = smem_lines[smem_y][smem_x2];
                            f = (f << 1) | (a > b);
                        }
                        dest(y, x) = f;
                    }
                }
                __syncthreads();
            }
        }
        }

        void censusTransform(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dest, cv::cuda::Stream& _stream)
        {
            CV_Assert(src.size() == dest.size());
            CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1);
            const int width_per_block = BLOCK_SIZE - WINDOW_WIDTH + 1;
            const int height_per_block = LINES_PER_BLOCK;
            const dim3 gdim(
                (src.cols  + width_per_block  - 1) / width_per_block,
                (src.rows + height_per_block - 1) / height_per_block);
            const dim3 bdim(BLOCK_SIZE);
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
            switch (src.type())
            {
            case CV_8UC1:
                census_transform_kernel<uint8_t><<<gdim, bdim, 0, stream>>>(src, dest);
                break;
            case CV_16UC1:
                census_transform_kernel<uint16_t><<<gdim, bdim, 0, stream>>>(src, dest);
                break;
            }
        }
    }
}}}

#endif /* CUDA_DISABLER */
