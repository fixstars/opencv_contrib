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
#include "stereosgm_horizontal_path_aggregation.hpp"
#include "stereosgm_vertical_path_aggregation.hpp"
#include "stereosgm_oblique_path_aggregation.hpp"

namespace cv { namespace cuda { namespace device
{
namespace stereosgm
{
template <size_t MAX_DISPARITY>
void pathAggregation(const GpuMat& left, const GpuMat& right, GpuMat& dest, int p1, int p2, Stream& stream)
{
    static const unsigned int NUM_PATHS = 8;
    CV_Assert(left.size() == right.size());
    CV_Assert(left.type() == right.type());
    CV_Assert(left.type() == CV_32SC1);

    stream.waitForCompletion();
    std::array<Stream, NUM_PATHS> streams;
    std::array<Event, NUM_PATHS> events;

    const Size size = left.size();
    const size_t buffer_step = size.width * size.height * MAX_DISPARITY;
    CV_Assert(dest.rows == 1 && buffer_step * NUM_PATHS == dest.cols);
    std::array<GpuMat, NUM_PATHS> subs;
    for (int i = 0; i < NUM_PATHS; ++i) {
        subs[i] = dest.colRange(i * buffer_step, (i + 1) * buffer_step);
    }
    aggregateUp2DownPath         <MAX_DISPARITY>(left, right, subs[0], p1, p2, streams[0]);
    aggregateDown2UpPath         <MAX_DISPARITY>(left, right, subs[1], p1, p2, streams[1]);
    aggregateLeft2RightPath      <MAX_DISPARITY>(left, right, subs[2], p1, p2, streams[2]);
    aggregateRight2LeftPath      <MAX_DISPARITY>(left, right, subs[3], p1, p2, streams[3]);
    aggregateUpleft2DownrightPath<MAX_DISPARITY>(left, right, subs[4], p1, p2, streams[4]);
    aggregateUpright2DownleftPath<MAX_DISPARITY>(left, right, subs[5], p1, p2, streams[5]);
    aggregateDownright2UpleftPath<MAX_DISPARITY>(left, right, subs[6], p1, p2, streams[6]);
    aggregateDownleft2UprightPath<MAX_DISPARITY>(left, right, subs[7], p1, p2, streams[7]);

    // synchronization
    for (int i = 0; i < NUM_PATHS; ++i)
    {
        events[i].record(streams[i]);
        stream.waitEvent(events[i]);
        streams[i].waitForCompletion();
    }
}

template void pathAggregation< 64>(const GpuMat& left, const GpuMat& right, GpuMat& dest, int p1, int p2, Stream& stream);
template void pathAggregation<128>(const GpuMat& left, const GpuMat& right, GpuMat& dest, int p1, int p2, Stream& stream);
}
}}}
