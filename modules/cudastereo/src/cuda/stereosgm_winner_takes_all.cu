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
#include "stereosgm_winner_takes_all.hpp"
#include "stereosgm_utility.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/cudev/warp/reduce.hpp"
#include "opencv2/cudev/warp/shuffle.hpp"

namespace cv { namespace cuda { namespace device
{
namespace stereosgm
{
namespace {
static constexpr unsigned int NUM_PATHS = 8u;

static constexpr unsigned int WARP_SIZE = 32u;
static constexpr unsigned int WARPS_PER_BLOCK = 8u;
static constexpr unsigned int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;

struct OpAnd
{
    __device__ __forceinline__ bool operator()(bool x, bool y) const
    {
        return x && y;
    }
};
static constexpr OpAnd OP_AND{};

struct OpMin
{
    __device__ __forceinline__ uint32_t operator()(uint32_t x, uint32_t y) const
    {
        return ::min(x, y);
    }
};
static constexpr OpMin OP_MIN{};

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

__device__ int unpack_index(uint32_t packed){
    return packed & 0xffffu;
}

using ComputeDisparity = uint32_t(*)(uint32_t, uint32_t, uint16_t*);

__device__ inline uint32_t compute_disparity_normal(uint32_t disp, uint32_t cost = 0, uint16_t* smem = nullptr)
{
    return disp;
}

template <size_t MAX_DISPARITY>
__device__ inline uint32_t compute_disparity_subpixel(uint32_t disp, uint32_t cost, uint16_t* smem)
{
    uint32_t subp = disp;
    subp <<= StereoMatcher::DISP_SHIFT;
    if (disp > 0 && disp < MAX_DISPARITY - 1)
    {
        const int left = smem[disp - 1];
        const int right = smem[disp + 1];
        const int numer = left - right;
        const int denom = left - 2 * cost + right;
        subp += ((numer << StereoMatcher::DISP_SHIFT) + denom) / (2 * denom);
    }
    return subp;
}


template <unsigned int MAX_DISPARITY, ComputeDisparity compute_disparity = compute_disparity_normal>
__global__ void winner_takes_all_kernel(
    const PtrStep<uint8_t> _src,
    PtrStep<int16_t> _left_dest,
    PtrStep<int16_t> _right_dest,
    int width,
    int height,
    float uniqueness)
{
    static const unsigned int ACCUMULATION_PER_THREAD = 16u;
    static const unsigned int REDUCTION_PER_THREAD = MAX_DISPARITY / WARP_SIZE;
    static const unsigned int ACCUMULATION_INTERVAL = ACCUMULATION_PER_THREAD / REDUCTION_PER_THREAD;
    static const unsigned int UNROLL_DEPTH =
        (REDUCTION_PER_THREAD > ACCUMULATION_INTERVAL)
            ? REDUCTION_PER_THREAD
            : ACCUMULATION_INTERVAL;

    const unsigned int cost_step = MAX_DISPARITY * width * height;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;

    const unsigned int y = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const PtrStep<uint8_t> src{(uint8_t*)&_src(0, y * MAX_DISPARITY * width), height * width * MAX_DISPARITY * NUM_PATHS};
    PtrStep<int16_t> left_dest{_left_dest.ptr(y), _left_dest.step};
    PtrStep<int16_t> right_dest{_right_dest.ptr(y), _right_dest.step};

    if(y >= height){
        return;
    }

    __shared__ uint16_t smem_cost_sum[WARPS_PER_BLOCK][ACCUMULATION_INTERVAL][MAX_DISPARITY];

    uint32_t right_best[REDUCTION_PER_THREAD];
    for(unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i){
        right_best[i] = 0xffffffffu;
    }

    for(unsigned int x0 = 0; x0 < width; x0 += UNROLL_DEPTH){
#pragma unroll
        for(unsigned int x1 = 0; x1 < UNROLL_DEPTH; ++x1){
            if(x1 % ACCUMULATION_INTERVAL == 0){
                const unsigned int k = lane_id * ACCUMULATION_PER_THREAD;
                const unsigned int k_hi = k / MAX_DISPARITY;
                const unsigned int k_lo = k % MAX_DISPARITY;
                const unsigned int x = x0 + x1 + k_hi;
                if(x < width){
                    const unsigned int offset = x * MAX_DISPARITY + k_lo;
                    uint32_t sum[ACCUMULATION_PER_THREAD];
                    for(unsigned int i = 0; i < ACCUMULATION_PER_THREAD; ++i){
                        sum[i] = 0;
                    }
                    for(unsigned int p = 0; p < NUM_PATHS; ++p){
                        uint32_t load_buffer[ACCUMULATION_PER_THREAD];
                        load_uint8_vector<ACCUMULATION_PER_THREAD>(
                            load_buffer, &src(0, p * cost_step + offset));
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
            if(x < width){
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
                uint32_t best = 0xffffffffu;
                for(unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i){
                    best = ::min(best, local_packed_cost[i]);
                }
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
                cv::cudev::warpReduce<uint32_t, OpMin>(nullptr, best, 0, OP_MIN);
                best = cv::cudev::shfl(best, 0);
            #else
                // TODO
            #endif
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
                    right_best[i] = ::min(right_best[i], recv);
                    if(d == MAX_DISPARITY - 1){
                        if(0 <= p){
                            right_dest(0, p) = compute_disparity_normal(unpack_index(right_best[i]));
                        }
                        right_best[i] = 0xffffffffu;
                    }
                }
                // Resume updating left to avoid execution dependency
                const uint32_t bestCost = unpack_cost(best);
                const int bestDisp = unpack_index(best);
                bool uniq = true;
                for(unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i){
                    const uint32_t x = local_packed_cost[i];
                    const bool uniq1 = unpack_cost(x) * uniqueness >= bestCost;
                    const bool uniq2 = ::abs(unpack_index(x) - bestDisp) <= 1;
                    uniq &= uniq1 || uniq2;
                }
                cv::cudev::warpReduce<bool, OpAnd>(nullptr, uniq, 0, OP_AND);
                if(lane_id == 0){
                    left_dest(0, x) = uniq ? compute_disparity(bestDisp, bestCost, smem_cost_sum[warp_id][smem_x]) : 0;
                }
            }
        }
    }
    for(unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i){
        const unsigned int k = lane_id * REDUCTION_PER_THREAD + i;
        const int p = static_cast<int>(((width - k) & ~(MAX_DISPARITY - 1)) + k);
        if(p < width){
            right_dest(0, p) = compute_disparity_normal(unpack_index(right_best[i]));
        }
    }
}
}

template <size_t MAX_DISPARITY>
void winnerTakesAll(const GpuMat& src, GpuMat& left, GpuMat& right, float uniqueness, bool subpixel, cv::cuda::Stream& _stream)
{
    cv::Size size = left.size();
    CV_Assert(src.rows == 1 && src.cols == size.width * size.height * MAX_DISPARITY * NUM_PATHS);
    CV_Assert(size == right.size());
    CV_Assert(left.type() == right.type());
    CV_Assert(src.type() == CV_8UC1);
    const int gdim =
        (size.height + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    if (subpixel) {
        winner_takes_all_kernel<MAX_DISPARITY, compute_disparity_subpixel<MAX_DISPARITY>><<<gdim, bdim, 0, stream>>>(
            src, left, right, size.width, size.height, uniqueness);
    } else {
        winner_takes_all_kernel<MAX_DISPARITY, compute_disparity_normal><<<gdim, bdim, 0, stream>>>(
            src, left, right, size.width, size.height, uniqueness);
    }
}
template CV_EXPORTS_W void winnerTakesAll< 64>(const GpuMat&, GpuMat&, GpuMat&, float, bool, cv::cuda::Stream&);
template CV_EXPORTS_W void winnerTakesAll<128>(const GpuMat&, GpuMat&, GpuMat&, float, bool, cv::cuda::Stream&);
}
}}}
