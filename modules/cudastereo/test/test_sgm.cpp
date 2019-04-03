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
    {
        cv::Mat src = readImage("stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
        cv::Mat expect;
        census_transform(src, expect);

        cv::cuda::GpuMat g_src, g_dst;
        g_src.upload(src);
        g_dst.create(src.size(), CV_32SC1);
        cv::cuda::device::stereosgm::censusTransform(g_src, g_dst, cv::cuda::Stream::Null());

        cv::Mat actual;
        g_dst.download(actual);

        EXPECT_MAT_NEAR(expect, actual, 1e-4);
    }

    INSTANTIATE_TEST_CASE_P(CUDA_StereoSGM_funcs, StereoSGM, ALL_DEVICES);
}} // namespace
#endif // HAVE_CUDA
