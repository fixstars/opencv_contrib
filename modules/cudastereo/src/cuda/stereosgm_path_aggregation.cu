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

#include "stereosgm_path_aggregation.hpp"
/*
#include "vertical_path_aggregation.hpp"
#include "oblique_path_aggregation.hpp"
*/
#include "stereosgm_horizontal_path_aggregation.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace stereosgm
    {
        template <size_t MAX_DISPARITY>
        void pathAggregation<MAX_DISPARITY>(const GpuMat& left, const GpuMat& right, GpuMat& dest, Stream& stream)
        {
            static const unsigned int NUM_PATHS = 8;
            CV_Assert(left.size() == right.size());
            CV_Assert(left.type() == right.type());
            CV_Assert(left.size() == dest.size());
            CV_Assert(left.type() == CV_32SC1);

            stream.waitForCompletion();
            std::array<Stream, NUM_PATHS> streams;
            std::array<Event, NUM_PATHS> events;

            // TODO add specific path aggregation
            const Size size = left.size();
            const size_t buffer_size = size.width * size.height * MAX_DISPARITY * NUM_PATHS;
            aggregateLeft2RightPath(left, right, dest.colRange(0 * buffer_size, 1 * buffer_size), p1, p2, stream);
            aggregateRight2LeftPath(left, right, dest.colRange(1 * buffer_size, 2 * buffer_size), p1, p2, stream);

            // synchronization
            for (int i = 0; i < NUM_PATHS; ++i)
            {
                events[i].record(streams[i]);
                stream.waitEvent(events[i]);
            }
        }

        template void pathAggregation< 64>(const GpuMat& left, const GpuMat& right, GpuMat& dest, Stream& stream);
        template void pathAggregation<128>(const GpuMat& left, const GpuMat& right, GpuMat& dest, Stream& stream);
	}
}}}
