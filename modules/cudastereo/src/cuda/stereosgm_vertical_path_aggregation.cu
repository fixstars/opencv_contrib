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
#include "stereosgm_vertical_path_aggregation.hpp"
#include "stereosgm_path_aggregation_common.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

namespace cv { namespace cuda { namespace device
{
namespace stereosgm
{
namespace
{
static constexpr unsigned int DP_BLOCK_SIZE = 16u;
static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

template <int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_vertical_path_kernel(
    PtrStep<int32_t> left,
    PtrStep<int32_t> right,
    PtrStep<uint8_t> dest,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

	static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
	static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

	static_assert(DIRECTION == 1 || DIRECTION == -1, "");
	if(width == 0 || height == 0){
		return;
	}

	__shared__ int32_t right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS + 1];
	DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp;

	const unsigned int warp_id  = threadIdx.x / WARP_SIZE;
	const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
	const unsigned int lane_id  = threadIdx.x % SUBGROUP_SIZE;
	const unsigned int shfl_mask =
		((1u << SUBGROUP_SIZE) - 1u) << (group_id * SUBGROUP_SIZE);

	const unsigned int x =
		blockIdx.x * PATHS_PER_BLOCK +
		warp_id    * PATHS_PER_WARP +
		group_id;
	const unsigned int right_x0 = blockIdx.x * PATHS_PER_BLOCK;
	const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

	const unsigned int right0_addr =
		(right_x0 + PATHS_PER_BLOCK - 1) - x + dp_offset;
	const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
	const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

	for(unsigned int iter = 0; iter < height; ++iter){
		const unsigned int y = (DIRECTION > 0 ? iter : height - 1 - iter);
		// Load left to register
		int32_t left_value;
		if(x < width){
			left_value = left(y, x);
		}
		// Load right to smem
		for(unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE){
			const unsigned int i = i0 + threadIdx.x;
			if(i < RIGHT_BUFFER_SIZE){
				const int x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i);
				int32_t right_value = 0;
				if(0 <= x && x < static_cast<int>(width)){
					right_value = right(y, x);
				}
				const unsigned int lo = i % DP_BLOCK_SIZE;
				const unsigned int hi = i / DP_BLOCK_SIZE;
				right_buffer[lo][hi] = right_value;
				if(hi > 0){
					right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
				}
			}
		}
		__syncthreads();
		// Compute
		if(x < width){
			int32_t right_values[DP_BLOCK_SIZE];
			for(unsigned int j = 0; j < DP_BLOCK_SIZE; ++j){
				right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
			}
			uint32_t local_costs[DP_BLOCK_SIZE];
			for(unsigned int j = 0; j < DP_BLOCK_SIZE; ++j){
				local_costs[j] = __popc(left_value ^ right_values[j]);
			}
			dp.update(local_costs, p1, p2, shfl_mask);
			store_uint8_vector<DP_BLOCK_SIZE>(
				&dest(0, dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width),
				dp.dp);
		}
		__syncthreads();
	}
}
}

template <unsigned int MAX_DISPARITY>
void aggregateUp2DownPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream _stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    const Size size = left.size();
	const int gdim = (size.width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
	aggregate_vertical_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		left, right, dest, size.width, size.height, p1, p2);
}

template <unsigned int MAX_DISPARITY>
void aggregateDown2UpPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream _stream)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    const Size size = left.size();
	const int gdim = (size.width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
	aggregate_vertical_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		left, right, dest, size.width, size.height, p1, p2);
}


template void aggregateUp2DownPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream stream);

template void aggregateUp2DownPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream stream);

template void aggregateDown2UpPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream stream);

template void aggregateDown2UpPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream stream);

}
}}}
