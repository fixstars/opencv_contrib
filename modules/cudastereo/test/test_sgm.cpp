/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include <limits>

#ifdef HAVE_CUDA

#ifdef _WIN32
#define popcnt64 __popcnt64
#else
#define popcnt64 __builtin_popcountll
#endif

namespace opencv_test { namespace {

    void census_transform(const cv::Mat& src, cv::Mat& dst)
    {
        const int hor = 9 / 2, ver = 7 / 2;
        dst.create(src.size(), CV_32SC1);
        dst = 0;
        for (int y = ver; y < static_cast<int>(src.rows) - ver; ++y) {
            for (int x = hor; x < static_cast<int>(src.cols) - hor; ++x) {
                const auto c = src.at<uint8_t>(y, x);
                int32_t value = 0;
                for (int dy = -ver; dy <= 0; ++dy) {
                    for (int dx = -hor; dx <= (dy == 0 ? -1 : hor); ++dx) {
                        const auto a = src.at<uint8_t>(y + dy, x + dx);
                        const auto b = src.at<uint8_t>(y - dy, x - dx);
                        value <<= 1;
                        if (a > b) { value |= 1; }
                    }
                }
                dst.at<int32_t>(y, x) = value;
            }
        }
    }

    PARAM_TEST_CASE(CensusTransformImage, cv::cuda::DeviceInfo, std::string, UseRoi)
    {
        cv::cuda::DeviceInfo devInfo;
        std::string path;
        bool useRoi;

        virtual void SetUp()
        {
            devInfo = GET_PARAM(0);
            path = GET_PARAM(1);
            useRoi = GET_PARAM(2);

            cv::cuda::setDevice(devInfo.deviceID());
        }
    };

    CUDA_TEST_P(CensusTransformImage, CensusTransformImageTest)
    {
        cv::Mat image = readImage(path, cv::IMREAD_GRAYSCALE);
        cv::Mat dst_gold;
        census_transform(image, dst_gold);

        cv::cuda::GpuMat g_dst;
        g_dst.create(image.size(), CV_32SC1);
        cv::cuda::device::stereosgm::censusTransform(loadMat(image, useRoi), g_dst, cv::cuda::Stream::Null());

        cv::Mat dst;
        g_dst.download(dst);

        EXPECT_MAT_NEAR(dst_gold, dst, 0);
    }

    INSTANTIATE_TEST_CASE_P(CUDA_StereoSGM_funcs, CensusTransformImage, testing::Combine(
        ALL_DEVICES,
        testing::Values("stereobm/aloe-L.png", "stereobm/aloe-R.png"),
        WHOLE_SUBMAT));

    PARAM_TEST_CASE(CensusTransformRandom, cv::cuda::DeviceInfo, cv::Size, UseRoi)
    {
        cv::cuda::DeviceInfo devInfo;
        cv::Size size;
        bool useRoi;

        virtual void SetUp()
        {
            devInfo = GET_PARAM(0);
            size = GET_PARAM(1);
            useRoi = GET_PARAM(2);

            cv::cuda::setDevice(devInfo.deviceID());
        }
    };

    CUDA_TEST_P(CensusTransformRandom, CensusTransformRandomTest)
    {
        cv::Mat image = randomMat(size, CV_8UC1);
        cv::Mat dst_gold;
        census_transform(image, dst_gold);

        cv::cuda::GpuMat g_dst;
        g_dst.create(image.size(), CV_32SC1);
        cv::cuda::device::stereosgm::censusTransform(loadMat(image, useRoi), g_dst, cv::cuda::Stream::Null());

        cv::Mat dst;
        g_dst.download(dst);

        EXPECT_MAT_NEAR(dst_gold, dst, 0);
    }

    INSTANTIATE_TEST_CASE_P(CUDA_StereoSGM_funcs, CensusTransformRandom, testing::Combine(
        ALL_DEVICES,
        DIFFERENT_SIZES,
        WHOLE_SUBMAT));

    static void path_aggregation(
        const cv::Mat& left,
        const cv::Mat& right,
        cv::Mat& dst,
        int max_disparity, int p1, int p2,
        int dx, int dy)
    {
        const int width = left.cols;
        const int height = left.rows;
        dst.create(cv::Size(width * height * max_disparity, 1), CV_8UC1);
        std::vector<int> before(max_disparity);
        for (int i = (dy < 0 ? height - 1 : 0); 0 <= i && i < height; i += (dy < 0 ? -1 : 1)) {
            for (int j = (dx < 0 ? width - 1 : 0); 0 <= j && j < width; j += (dx < 0 ? -1 : 1)) {
                const int i2 = i - dy, j2 = j - dx;
                const bool inside = (0 <= i2 && i2 < height && 0 <= j2 && j2 < width);
                for (int k = 0; k < max_disparity; ++k) {
                    before[k] = inside ? dst.at<uint8_t>(0, k + (j2 + i2 * width) * max_disparity) : 0;
                }
                const int min_cost = *min_element(before.begin(), before.end());
                for (int k = 0; k < max_disparity; ++k) {
                    const auto l = left.at<int32_t>(i, j);
                    const auto r = (k > j ? 0 : right.at<int32_t>(i, j - k));
                    int cost = std::min(before[k] - min_cost, p2);
                    if (k > 0) {
                        cost = std::min(cost, before[k - 1] - min_cost + p1);
                    }
                    if (k + 1 < max_disparity) {
                        cost = std::min(cost, before[k + 1] - min_cost + p1);
                    }
                    cost += static_cast<int>(popcnt64(l ^ r));
                    dst.at<uint8_t>(0, k + (j + i * width) * max_disparity) = static_cast<uint8_t>(cost);
                }
            }
        }
    }

    static constexpr size_t DISPARITY = 128;
    static constexpr int P1 = 10;
    static constexpr int P2 = 120;

    PARAM_TEST_CASE(StereoSGM_PathAggregation, cv::cuda::DeviceInfo, cv::Size, UseRoi)
    {
        cv::cuda::DeviceInfo devInfo;
        cv::Size size;
        bool useRoi;

        virtual void SetUp()
        {
            devInfo = GET_PARAM(0);
            size = GET_PARAM(1);
            useRoi = GET_PARAM(2);

            cv::cuda::setDevice(devInfo.deviceID());
        }

        template<typename T>
        void test_path_aggregation(T func, int dx, int dy)
        {
            cv::Mat left_image = randomMat(size, CV_32SC1, 0.0, static_cast<double>(std::numeric_limits<int32_t>::max()));
            cv::Mat right_image = randomMat(size, CV_32SC1, 0.0, static_cast<double>(std::numeric_limits<int32_t>::max()));
            cv::Mat dst_gold;
            path_aggregation(left_image, right_image, dst_gold, DISPARITY, P1, P2, dx, dy);

            cv::cuda::GpuMat g_dst;
            g_dst.create(cv::Size(left_image.cols * left_image.rows * DISPARITY, 1), CV_8UC1);
            func(loadMat(left_image, useRoi), loadMat(right_image, useRoi), g_dst, P1, P2, cv::cuda::Stream::Null());

            cv::Mat dst;
            g_dst.download(dst);

            EXPECT_MAT_NEAR(dst_gold, dst, 0);
        }
    };

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomLeft2Right)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::aggregateLeft2RightPath<DISPARITY>, 1, 0);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomRight2Left)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::aggregateRight2LeftPath<DISPARITY>, -1, 0);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomUp2Down)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::aggregateUp2DownPath<DISPARITY>, 0, 1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomDown2Up)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::aggregateDown2UpPath<DISPARITY>, 0, -1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomUpLeft2DownRight)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::aggregateUpleft2DownrightPath<DISPARITY>, 1, 1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomUpRight2DownLeft)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::aggregateUpright2DownleftPath<DISPARITY>, -1, 1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomDownRight2UpLeft)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::aggregateDownright2UpleftPath<DISPARITY>, -1, -1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomDownLeft2UpRight)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::aggregateDownleft2UprightPath<DISPARITY>, 1, -1);
    }

    INSTANTIATE_TEST_CASE_P(CUDA_StereoSGM_funcs, StereoSGM_PathAggregation, testing::Combine(
        ALL_DEVICES,
        DIFFERENT_SIZES,
        WHOLE_SUBMAT));


    static constexpr size_t NUM_PATHS = 8;

    void winner_takes_all_left(
        const cv::Mat& src,
        cv::Mat& dst,
        size_t width, size_t height, size_t disparity,
        float uniqueness, bool subpixel)
    {
        dst.create(cv::Size(width, height), CV_16UC1);
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                std::vector<std::pair<int, int>> v;
                for (size_t k = 0; k < disparity; ++k) {
                    int cost_sum = 0;
                    for (size_t p = 0; p < NUM_PATHS; ++p) {
                        cost_sum += static_cast<int>(src.at<uint8_t>(0,
                            p * disparity * width * height +
                                i * disparity * width +
                                j * disparity +
                                k));
                    }
                    v.emplace_back(cost_sum, static_cast<int>(k));
                }
                auto w = v;
                sort(v.begin(), v.end());
                if (v.size() < 2) {
                    dst.at<uint16_t>(i, j) = 0;
                }
                else
                {
                    const int cost0 = v[0].first;
                    const int cost1 = v[1].first;
                    const int disp0 = v[0].second;
                    const int disp1 = v[1].second;
                    uint16_t disp = 0;
                    if (cost1 * uniqueness < cost0 && abs(disp0 - disp1) > 1) {
                        dst = 0;
                    }
                    else {
                        disp = disp0;
                        if (subpixel) {
                            disp <<= cv::StereoMatcher::DISP_SHIFT;
                            if (0 < disp0 && disp0 < static_cast<int>(disparity) - 1) {
                                const int left = w[disp0 - 1].first;
                                const int right = w[disp0 + 1].first;
                                const int numer = left - right;
                                const int denom = left - 2 * cost0 + right;
                                disp += ((numer << cv::StereoMatcher::DISP_SHIFT) + denom) / (2 * denom);
                            }
                        }
                    }
                    dst.at<uint16_t>(i, j) = disp;
                }
            }
        }
    }

    PARAM_TEST_CASE(StereoSGM_WinnerTakesAll, cv::cuda::DeviceInfo, cv::Size)
    {
        cv::cuda::DeviceInfo devInfo;
        cv::Size size;

        virtual void SetUp()
        {
            devInfo = GET_PARAM(0);
            size = GET_PARAM(1);

            cv::cuda::setDevice(devInfo.deviceID());
        }
    };

    CUDA_TEST_P(StereoSGM_WinnerTakesAll, RandomLeft)
    {
        cv::Mat aggregated = randomMat(cv::Size(size.width * size.height * DISPARITY * NUM_PATHS, 1), CV_8UC1, 0.0, static_cast<double>(std::numeric_limits<uint8_t>::max()));
        cv::Mat dst_gold;
        winner_takes_all_left(aggregated, dst_gold, size.width, size.height, DISPARITY, 0.95f, true);

        cv::cuda::GpuMat g_src, g_dst, g_dst_right;
        g_src.upload(aggregated);
        g_dst.create(size, CV_16UC1);
        g_dst_right.create(size, CV_16UC1);
        cv::cuda::device::stereosgm::winnerTakesAll<DISPARITY>(g_src, g_dst, g_dst_right, 0.95f, true, cv::cuda::Stream::Null());

        cv::Mat dst;
        g_dst.download(dst);

        EXPECT_MAT_NEAR(dst_gold, dst, 0);
    }

    INSTANTIATE_TEST_CASE_P(CUDA_StereoSGM_funcs, StereoSGM_WinnerTakesAll, testing::Combine(
        ALL_DEVICES,
        DIFFERENT_SIZES));

}} // namespace
#endif // HAVE_CUDA
