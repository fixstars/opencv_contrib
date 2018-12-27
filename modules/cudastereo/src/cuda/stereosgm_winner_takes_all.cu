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

#include <cstdio>
#include <cuda.h>
#include <device_functions.h>
#include "stereosgm_winner_takes_all.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace stereosgm
    {
        namespace {
            template <typename T, typename S>
            __device__ inline T load_as(const S *p){
                return *reinterpret_cast<const T *>(p);
            }

            template <typename T, typename S>
            __device__ inline void store_as(S *p, const T& x){
                *reinterpret_cast<T *>(p) = x;
            }

            template <typename T>
            __device__ inline uint32_t pack_uint8x4(T x, T y, T z, T w){
                uchar4 uint8x4;
                uint8x4.x = static_cast<uint8_t>(x);
                uint8x4.y = static_cast<uint8_t>(y);
                uint8x4.z = static_cast<uint8_t>(z);
                uint8x4.w = static_cast<uint8_t>(w);
                return load_as<uint32_t>(&uint8x4);
            }

            template <unsigned int N>
            __device__ inline void load_uint8_vector(uint32_t *dest, const uint8_t *ptr);

            template <>
            __device__ inline void load_uint8_vector<1u>(uint32_t *dest, const uint8_t *ptr){
                dest[0] = static_cast<uint32_t>(ptr[0]);
            }

            template <>
            __device__ inline void load_uint8_vector<2u>(uint32_t *dest, const uint8_t *ptr){
                const auto uint8x2 = load_as<uchar2>(ptr);
                dest[0] = uint8x2.x; dest[1] = uint8x2.y;
            }

            template <>
            __device__ inline void load_uint8_vector<4u>(uint32_t *dest, const uint8_t *ptr){
                const auto uint8x4 = load_as<uchar4>(ptr);
                dest[0] = uint8x4.x; dest[1] = uint8x4.y; dest[2] = uint8x4.z; dest[3] = uint8x4.w;
            }

            template <>
            __device__ inline void load_uint8_vector<8u>(uint32_t *dest, const uint8_t *ptr){
                const auto uint32x2 = load_as<uint2>(ptr);
                load_uint8_vector<4u>(dest + 0, reinterpret_cast<const uint8_t *>(&uint32x2.x));
                load_uint8_vector<4u>(dest + 4, reinterpret_cast<const uint8_t *>(&uint32x2.y));
            }

            template <>
            __device__ inline void load_uint8_vector<16u>(uint32_t *dest, const uint8_t *ptr){
                const auto uint32x4 = load_as<uint4>(ptr);
                load_uint8_vector<4u>(dest +  0, reinterpret_cast<const uint8_t *>(&uint32x4.x));
                load_uint8_vector<4u>(dest +  4, reinterpret_cast<const uint8_t *>(&uint32x4.y));
                load_uint8_vector<4u>(dest +  8, reinterpret_cast<const uint8_t *>(&uint32x4.z));
                load_uint8_vector<4u>(dest + 12, reinterpret_cast<const uint8_t *>(&uint32x4.w));
            }


            template <unsigned int N>
            __device__ inline void store_uint8_vector(uint8_t *dest, const uint32_t *ptr);

            template <>
            __device__ inline void store_uint8_vector<1u>(uint8_t *dest, const uint32_t *ptr){
                dest[0] = static_cast<uint8_t>(ptr[0]);
            }

            template <>
            __device__ inline void store_uint8_vector<2u>(uint8_t *dest, const uint32_t *ptr){
                uchar2 uint8x2;
                uint8x2.x = static_cast<uint8_t>(ptr[0]);
                uint8x2.y = static_cast<uint8_t>(ptr[0]);
                store_as<uchar2>(dest, uint8x2);
            }

            template <>
            __device__ inline void store_uint8_vector<4u>(uint8_t *dest, const uint32_t *ptr){
                store_as<uint32_t>(dest, pack_uint8x4(ptr[0], ptr[1], ptr[2], ptr[3]));
            }

            template <>
            __device__ inline void store_uint8_vector<8u>(uint8_t *dest, const uint32_t *ptr){
                uint2 uint32x2;
                uint32x2.x = pack_uint8x4(ptr[0], ptr[1], ptr[2], ptr[3]);
                uint32x2.y = pack_uint8x4(ptr[4], ptr[5], ptr[6], ptr[7]);
                store_as<uint2>(dest, uint32x2);
            }

            template <>
            __device__ inline void store_uint8_vector<16u>(uint8_t *dest, const uint32_t *ptr){
                uint4 uint32x4;
                uint32x4.x = pack_uint8x4(ptr[ 0], ptr[ 1], ptr[ 2], ptr[ 3]);
                uint32x4.y = pack_uint8x4(ptr[ 4], ptr[ 5], ptr[ 6], ptr[ 7]);
                uint32x4.z = pack_uint8x4(ptr[ 8], ptr[ 9], ptr[10], ptr[11]);
                uint32x4.w = pack_uint8x4(ptr[12], ptr[13], ptr[14], ptr[15]);
                store_as<uint4>(dest, uint32x4);
            }


            template <unsigned int N>
            __device__ inline void load_uint16_vector(uint32_t *dest, const uint16_t *ptr);

            template <>
            __device__ inline void load_uint16_vector<1u>(uint32_t *dest, const uint16_t *ptr){
                dest[0] = static_cast<uint32_t>(ptr[0]);
            }

            template <>
            __device__ inline void load_uint16_vector<2u>(uint32_t *dest, const uint16_t *ptr){
                const auto uint16x2 = load_as<ushort2>(ptr);
                dest[0] = uint16x2.x; dest[1] = uint16x2.y;
            }

            template <>
            __device__ inline void load_uint16_vector<4u>(uint32_t *dest, const uint16_t *ptr){
                const auto uint16x4 = load_as<ushort4>(ptr);
                dest[0] = uint16x4.x; dest[1] = uint16x4.y; dest[2] = uint16x4.z; dest[3] = uint16x4.w;
            }

            template <>
            __device__ inline void load_uint16_vector<8u>(uint32_t *dest, const uint16_t *ptr){
                const auto uint32x4 = load_as<uint4>(ptr);
                load_uint16_vector<2u>(dest + 0, reinterpret_cast<const uint16_t *>(&uint32x4.x));
                load_uint16_vector<2u>(dest + 2, reinterpret_cast<const uint16_t *>(&uint32x4.y));
                load_uint16_vector<2u>(dest + 4, reinterpret_cast<const uint16_t *>(&uint32x4.z));
                load_uint16_vector<2u>(dest + 6, reinterpret_cast<const uint16_t *>(&uint32x4.w));
            }


            template <unsigned int N>
            __device__ inline void store_uint16_vector(uint16_t *dest, const uint32_t *ptr);

            template <>
            __device__ inline void store_uint16_vector<1u>(uint16_t *dest, const uint32_t *ptr){
                dest[0] = static_cast<uint16_t>(ptr[0]);
            }

            template <>
            __device__ inline void store_uint16_vector<2u>(uint16_t *dest, const uint32_t *ptr){
                ushort2 uint16x2;
                uint16x2.x = static_cast<uint16_t>(ptr[0]);
                uint16x2.y = static_cast<uint16_t>(ptr[1]);
                store_as<ushort2>(dest, uint16x2);
            }

            template <>
            __device__ inline void store_uint16_vector<4u>(uint16_t *dest, const uint32_t *ptr){
                ushort4 uint16x4;
                uint16x4.x = static_cast<uint16_t>(ptr[0]);
                uint16x4.y = static_cast<uint16_t>(ptr[1]);
                uint16x4.z = static_cast<uint16_t>(ptr[2]);
                uint16x4.w = static_cast<uint16_t>(ptr[3]);
                store_as<ushort4>(dest, uint16x4);
            }

            template <>
            __device__ inline void store_uint16_vector<8u>(uint16_t *dest, const uint32_t *ptr){
                uint4 uint32x4;
                store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.x), &ptr[0]);
                store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.y), &ptr[2]);
                store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.z), &ptr[4]);
                store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.w), &ptr[6]);
                store_as<uint4>(dest, uint32x4);
            }

            template <>
            __device__ inline void store_uint16_vector<16u>(uint16_t *dest, const uint32_t *ptr){
                store_uint16_vector<8u>(dest + 0, ptr + 0);
                store_uint16_vector<8u>(dest + 8, ptr + 8);
            }

            static constexpr unsigned int NUM_PATHS = 8u;

            static constexpr unsigned int WARP_SIZE = 32u;
            static constexpr unsigned int WARPS_PER_BLOCK = 8u;
            static constexpr unsigned int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;


            __device__ inline void update_top2(uint32_t& v0, uint32_t& v1, uint32_t x){
                const uint32_t y = umax(x, v0);
                v0 = umin(x, v0);
                v1 = umin(y, v1);
            }

            struct Top2 {
                uint32_t values[2];

                __device__ void initialize(){
                    values[0] = 0xffffffffu;
                    values[1] = 0xffffffffu;
                }

                __device__ void push(uint32_t x){
                    update_top2(values[0], values[1], x);
                }
            };

            template <unsigned int GROUP_SIZE, unsigned int STEP>
            struct subgroup_merge_top2_impl {
                static __device__ Top2 call(Top2 x){
            #if CUDA_VERSION >= 9000
                    const uint32_t a = __shfl_xor_sync(0xffffffffu, x.values[0], STEP / 2, GROUP_SIZE);
                    const uint32_t b = __shfl_xor_sync(0xffffffffu, x.values[1], STEP / 2, GROUP_SIZE);
            #else
                    const uint32_t a = __shfl_xor(x.values[0], STEP / 2, GROUP_SIZE);
                    const uint32_t b = __shfl_xor(x.values[1], STEP / 2, GROUP_SIZE);
            #endif
                    x.push(a);
                    x.push(b);
                    return subgroup_merge_top2_impl<GROUP_SIZE, STEP / 2>::call(x);
                }
            };

            template <unsigned int GROUP_SIZE>
            struct subgroup_merge_top2_impl<GROUP_SIZE, 1u> {
                static __device__ Top2 call(Top2 x){
                    return x;
                }
            };

            template <unsigned int GROUP_SIZE>
            __device__ inline Top2 subgroup_merge_top2(Top2 x){
                return subgroup_merge_top2_impl<GROUP_SIZE, GROUP_SIZE>::call(x);
            }

            __device__ inline uint32_t pack_cost_index(uint32_t cost, uint32_t index){
                union {
                    uint32_t uint32;
                    ushort2 uint16x2;
                } u;
                u.uint16x2.x = static_cast<uint16_t>(index);
                u.uint16x2.y = static_cast<uint16_t>(cost);
                return u.uint32;
            }

            __device__ uint32_t unpack_cost(uint32_t packed){
                return packed >> 16;
            }

            __device__ uint32_t unpack_index(uint32_t packed){
                return packed & 0xffffu;
            }

            using ComputeDisparity = uint32_t(*)(Top2, float, uint16_t*);

            template <size_t MAX_DISPARITY>
            __device__ inline uint32_t compute_disparity_normal(Top2 t2, float uniqueness, uint16_t* smem = nullptr)
            {
                const float cost0 = static_cast<float>(unpack_cost(t2.values[0]));
                const float cost1 = static_cast<float>(unpack_cost(t2.values[1]));
                const int disp0 = static_cast<int>(unpack_index(t2.values[0]));
                const int disp1 = static_cast<int>(unpack_index(t2.values[1]));
                if(cost1 * uniqueness >= cost0){
                    return disp0;
                }else if(abs(disp1 - disp0) <= 1){
                    return disp0;
                }else{
                    return 0;
                }
            }

            template <size_t MAX_DISPARITY>
            __device__ inline uint32_t compute_disparity_subpixel(Top2 t2, float uniqueness, uint16_t* smem)
            {
                const float cost0 = static_cast<float>(unpack_cost(t2.values[0]));
                const float cost1 = static_cast<float>(unpack_cost(t2.values[1]));
                const int disp0 = static_cast<int>(unpack_index(t2.values[0]));
                const int disp1 = static_cast<int>(unpack_index(t2.values[1]));
                if(cost1 * uniqueness >= cost0
                    || abs(disp1 - disp0) <= 1){
                    int disp = disp0;
                    disp <<= DISP_SHIFT;
                    if (disp0 > 0 && disp0 < MAX_DISPARITY - 1) {
                        const int numer = smem[disp0 - 1] - smem[disp0 + 1];
                        const int denom = smem[disp0 - 1] - 2 * smem[disp0] + smem[disp0 + 1];
                        disp += ((numer << DISP_SHIFT) + denom) / (2 * denom);
                    }
                    return disp;
                }else{
                    return 0;
                }
            }


            template <unsigned int MAX_DISPARITY, ComputeDisparity compute_disparity = compute_disparity_normal<MAX_DISPARITY>>
            __global__ void winner_takes_all_kernel(
                const PtrStepSz<uint8_t> _src,
                PtrStep<uint16_t> _left_dest,
                PtrStep<uint16_t> _right_dest,
                float uniqueness)
            {
                static const unsigned int ACCUMULATION_PER_THREAD = 16u;
                static const unsigned int REDUCTION_PER_THREAD = MAX_DISPARITY / WARP_SIZE;
                static const unsigned int ACCUMULATION_INTERVAL = ACCUMULATION_PER_THREAD / REDUCTION_PER_THREAD;
                static const unsigned int UNROLL_DEPTH =
                    (REDUCTION_PER_THREAD > ACCUMULATION_INTERVAL)
                        ? REDUCTION_PER_THREAD
                        : ACCUMULATION_INTERVAL;

                const unsigned int cost_step = MAX_DISPARITY * _src.cols * _src.rows;
                const unsigned int warp_id = threadIdx.x / WARP_SIZE;
                const unsigned int lane_id = threadIdx.x % WARP_SIZE;

                const unsigned int y = blockIdx.x * WARPS_PER_BLOCK + warp_id;
                const uint8_t *src = _src.ptr(y * MAX_DISPARITY);
                uint16_t *left_dest  = _left_dest.ptr(y);
                uint16_t *right_dest  = _right_dest.ptr(y);

                if(y >= _src.rows){
                    return;
                }

                __shared__ uint16_t smem_cost_sum[WARPS_PER_BLOCK][ACCUMULATION_INTERVAL][MAX_DISPARITY];

                Top2 right_top2[REDUCTION_PER_THREAD];
                for(unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i){
                    right_top2[i].initialize();
                }

                for(unsigned int x0 = 0; x0 < _src.cols; x0 += UNROLL_DEPTH){
            #pragma unroll
                    for(unsigned int x1 = 0; x1 < UNROLL_DEPTH; ++x1){
                        if(x1 % ACCUMULATION_INTERVAL == 0){
                            const unsigned int k = lane_id * ACCUMULATION_PER_THREAD;
                            const unsigned int k_hi = k / MAX_DISPARITY;
                            const unsigned int k_lo = k % MAX_DISPARITY;
                            const unsigned int x = x0 + x1 + k_hi;
                            if(x < _src.cols){
                                const unsigned int offset = x * MAX_DISPARITY + k_lo;
                                uint32_t sum[ACCUMULATION_PER_THREAD];
                                for(unsigned int i = 0; i < ACCUMULATION_PER_THREAD; ++i){
                                    sum[i] = 0;
                                }
                                for(unsigned int p = 0; p < NUM_PATHS; ++p){
                                    uint32_t load_buffer[ACCUMULATION_PER_THREAD];
                                    load_uint8_vector<ACCUMULATION_PER_THREAD>(
                                        load_buffer, &src[p * cost_step + offset]);
                                    for(unsigned int i = 0; i < ACCUMULATION_PER_THREAD; ++i){
                                        sum[i] += load_buffer[i];
                                    }
                                }
                                store_uint16_vector<ACCUMULATION_PER_THREAD>(
                                    &smem_cost_sum[warp_id][k_hi][k_lo], sum);
                            }
            #if CUDA_VERSION >= 9000
                            __syncwarp();
            #else
                            __threadfence_block();
            #endif
                        }
                        const unsigned int x = x0 + x1;
                        if(x < _src.cols){
                            // Load sum of costs
                            const unsigned int smem_x = x1 % ACCUMULATION_INTERVAL;
                            const unsigned int k0 = lane_id * REDUCTION_PER_THREAD;
                            uint32_t local_cost_sum[REDUCTION_PER_THREAD];
                            load_uint16_vector<REDUCTION_PER_THREAD>(
                                local_cost_sum, &smem_cost_sum[warp_id][smem_x][k0]);
                            // Pack sum of costs and dispairty
                            uint32_t local_packed_cost[REDUCTION_PER_THREAD];
                            for(unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i){
                                local_packed_cost[i] = pack_cost_index(local_cost_sum[i], k0 + i);
                            }
                            // Update left
                            Top2 left_top2;
                            left_top2.initialize();
                            for(unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i){
                                left_top2.push(local_packed_cost[i]);
                            }
                            left_top2 = subgroup_merge_top2<WARP_SIZE>(left_top2);
                            if(lane_id == 0){
                                left_dest[x] = compute_disparity(left_top2, uniqueness, smem_cost_sum[warp_id][smem_x]);
                            }
                            // Update right
            #pragma unroll
                            for(unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i){
                                const unsigned int k = lane_id * REDUCTION_PER_THREAD + i;
                                const int p = static_cast<int>(((x - k) & ~(MAX_DISPARITY - 1)) + k);
                                const unsigned int d = static_cast<unsigned int>(x - p);
            #if CUDA_VERSION >= 9000
                                const uint32_t recv = __shfl_sync(0xffffffffu,
                                    local_packed_cost[(REDUCTION_PER_THREAD - i + x1) % REDUCTION_PER_THREAD],
                                    d / REDUCTION_PER_THREAD,
                                    WARP_SIZE);
            #else
                                const uint32_t recv = __shfl(
                                    local_packed_cost[(REDUCTION_PER_THREAD - i + x1) % REDUCTION_PER_THREAD],
                                    d / REDUCTION_PER_THREAD,
                                    WARP_SIZE);
            #endif
                                right_top2[i].push(recv);
                                if(d == MAX_DISPARITY - 1){
                                    if(0 <= p){
                                        right_dest[p] = compute_disparity_normal<MAX_DISPARITY>(right_top2[i], uniqueness);
                                    }
                                    right_top2[i].initialize();
                                }
                            }
                        }
                    }
                }
                for(unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i){
                    const unsigned int k = lane_id * REDUCTION_PER_THREAD + i;
                    const int p = static_cast<int>(((_src.cols - k) & ~(MAX_DISPARITY - 1)) + k);
                    if(p < _src.cols){
                        right_dest[p] = compute_disparity_normal<MAX_DISPARITY>(right_top2[i], uniqueness);
                    }
                }
            }
        }

        template <size_t MAX_DISPARITY>
        void winnerTakesAll(const GpuMat& src, GpuMat& left, GpuMat& right, cv::cuda::Stream& _stream)
        {
            CV_Assert(src.size() == left.size() && left.size() == right.size());
            CV_Assert(src.type() == CV_16UC1);
            const int gdim =
                (src.rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            const int bdim = BLOCK_SIZE;
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
            if (subpixel) {
                winner_takes_all_kernel<MAX_DISPARITY, compute_disparity_subpixel<MAX_DISPARITY>><<<gdim, bdim, 0, stream>>>(
                    src, left_dest, right_dest, uniqueness);
            } else {
                winner_takes_all_kernel<MAX_DISPARITY, compute_disparity_normal<MAX_DISPARITY>><<<gdim, bdim, 0, stream>>>(
                    src, left_dest, right_dest, uniqueness);
            }
        }
    }
}}}
