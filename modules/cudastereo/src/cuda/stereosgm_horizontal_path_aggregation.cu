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
#include "stereosgm_horizontal_path_aggregation.hpp"
#include "stereosgm_path_aggregation_common.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

namespace cv { namespace cuda { namespace device
{
namespace stereosgm
{
namespace
{
static constexpr unsigned int DP_BLOCK_SIZE = 8u;
static constexpr unsigned int DP_BLOCKS_PER_THREAD = 1u;

static constexpr unsigned int WARPS_PER_BLOCK = 4u;
static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;


template <int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_horizontal_path_kernel(
    PtrStep<int32_t> left,
    PtrStep<int32_t> right,
    PtrStep<uint8_t> dest,
    int width,
    int height,
	unsigned int p1,
	unsigned int p2)
{
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int SUBGROUPS_PER_WARP = WARP_SIZE / SUBGROUP_SIZE;
	static const unsigned int PATHS_PER_WARP =
		WARP_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;
	static const unsigned int PATHS_PER_BLOCK =
		BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

	static_assert(DIRECTION == 1 || DIRECTION == -1, "");
	if(width == 0 || height == 0){
		return;
	}

	int32_t right_buffer[DP_BLOCKS_PER_THREAD][DP_BLOCK_SIZE];
	DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE> dp[DP_BLOCKS_PER_THREAD];

	const unsigned int warp_id  = threadIdx.x / WARP_SIZE;
	const unsigned int group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE;
	const unsigned int lane_id  = threadIdx.x % SUBGROUP_SIZE;
	const unsigned int shfl_mask =
		((1u << SUBGROUP_SIZE) - 1u) << (group_id * SUBGROUP_SIZE);

	const unsigned int y0 =
		PATHS_PER_BLOCK * blockIdx.x +
		PATHS_PER_WARP  * warp_id +
		group_id;
	const unsigned int feature_step = SUBGROUPS_PER_WARP;
	const unsigned int dest_step = SUBGROUPS_PER_WARP * MAX_DISPARITY * width;
	const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;
	left  = PtrStep<int32_t>( left.ptr(y0),  left.step);
	right = PtrStep<int32_t>(right.ptr(y0), right.step);
    dest  = PtrStep<uint8_t>(&dest(0, y0 * width * MAX_DISPARITY), dest.step);

	if(y0 >= height){
		return;
	}

	if(DIRECTION > 0){
		for(unsigned int i = 0; i < DP_BLOCKS_PER_THREAD; ++i){
			for(unsigned int j = 0; j < DP_BLOCK_SIZE; ++j){
				right_buffer[i][j] = 0;
			}
		}
	}else{
		for(unsigned int i = 0; i < DP_BLOCKS_PER_THREAD; ++i){
			for(unsigned int j = 0; j < DP_BLOCK_SIZE; ++j){
				const int x = static_cast<int>(width - (j + dp_offset));
				if(0 <= x && x < static_cast<int>(width)){
					right_buffer[i][j] = __ldg(&right(i * feature_step, x));
				}else{
					right_buffer[i][j] = 0;
				}
			}
		}
	}

	int x0 = (DIRECTION > 0) ? 0 : static_cast<int>((width - 1) & ~(DP_BLOCK_SIZE - 1));
	for(unsigned int iter = 0; iter < width; iter += DP_BLOCK_SIZE){
		for(unsigned int i = 0; i < DP_BLOCK_SIZE; ++i){
			const unsigned int x = x0 + (DIRECTION > 0 ? i : (DP_BLOCK_SIZE - 1 - i));
			if(x >= width){
				continue;
			}
			for(unsigned int j = 0; j < DP_BLOCKS_PER_THREAD; ++j){
				const unsigned int y = y0 + j * SUBGROUPS_PER_WARP;
				if(y >= height){
					continue;
				}
				const int32_t left_value = __ldg(&left(j * feature_step, x));
				if(DIRECTION > 0){
					const int32_t t = right_buffer[j][DP_BLOCK_SIZE - 1];
					for(unsigned int k = DP_BLOCK_SIZE - 1; k > 0; --k){
						right_buffer[j][k] = right_buffer[j][k - 1];
					}
#if CUDA_VERSION >= 9000
					right_buffer[j][0] = __shfl_up_sync(shfl_mask, t, 1, SUBGROUP_SIZE);
#else
					right_buffer[j][0] = __shfl_up(t, 1, SUBGROUP_SIZE);
#endif
					if(lane_id == 0){
						right_buffer[j][0] =
							__ldg(&right(j * feature_step, x - dp_offset));
					}
				}else{
					const int32_t t = right_buffer[j][0];
					for(unsigned int k = 1; k < DP_BLOCK_SIZE; ++k){
						right_buffer[j][k - 1] = right_buffer[j][k];
					}
#if CUDA_VERSION >= 9000
					right_buffer[j][DP_BLOCK_SIZE - 1] =
						__shfl_down_sync(shfl_mask, t, 1, SUBGROUP_SIZE);
#else
					right_buffer[j][DP_BLOCK_SIZE - 1] = __shfl_down(t, 1, SUBGROUP_SIZE);
#endif
					if(lane_id + 1 == SUBGROUP_SIZE){
						if(x >= dp_offset + DP_BLOCK_SIZE - 1){
							right_buffer[j][DP_BLOCK_SIZE - 1] =
								__ldg(&right(j * feature_step, x - (dp_offset + DP_BLOCK_SIZE - 1)));
						}else{
							right_buffer[j][DP_BLOCK_SIZE - 1] = 0;
						}
					}
				}
				uint32_t local_costs[DP_BLOCK_SIZE];
				for(unsigned int k = 0; k < DP_BLOCK_SIZE; ++k){
					local_costs[k] = __popc(left_value ^ right_buffer[j][k]);
				}
				dp[j].update(local_costs, p1, p2, shfl_mask);
				store_uint8_vector<DP_BLOCK_SIZE>(
					&dest(0, j * dest_step + x * MAX_DISPARITY + dp_offset),
					dp[j].dp);
			}
		}
		x0 += static_cast<int>(DP_BLOCK_SIZE) * DIRECTION;
	}
}
}


template <unsigned int MAX_DISPARITY>
void aggregateLeft2RightPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream& _stream)
{
    CV_Assert(left.size() == right.size());
    CV_Assert(left.size() == dest.size());
    CV_Assert(left.type() == right.type());
    CV_Assert(left.type() == CV_32SC1);
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK =
		BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

	const int gdim = (left.rows + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
	aggregate_horizontal_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		left, right, dest, left.cols, left.rows, p1, p2);
}

template <unsigned int MAX_DISPARITY>
void aggregateRight2LeftPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream& _stream)
{
    CV_Assert(left.size() == right.size());
    CV_Assert(left.size() == dest.size());
    CV_Assert(left.type() == right.type());
    CV_Assert(left.type() == CV_32SC1);
	static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
	static const unsigned int PATHS_PER_BLOCK =
		BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

	const int gdim = (left.rows + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
	const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
	aggregate_horizontal_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		left, right, dest, left.cols, left.rows, p1, p2);
}


template void aggregateLeft2RightPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream& _stream);

template void aggregateLeft2RightPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream& _stream);

template void aggregateRight2LeftPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream& _stream);

template void aggregateRight2LeftPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
	unsigned int p1,
	unsigned int p2,
	Stream& _stream);

}
}}}
