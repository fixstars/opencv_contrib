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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

Ptr<cuda::StereoSGM> cv::cuda::createStereoSGM(int, int, int, int) { throw_no_cuda(); return Ptr<cuda::StereoSGM>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace stereosgm
    {
        void censusTransform(const GpuMat& src, GpuMat& dest, cv::cuda::Stream& stream);
        template <size_t MAX_DISPARITY, size_t NUM_PATHS>
        void pathAggregation(const GpuMat& left, const GpuMat& right, GpuMat& dest, int p1, int p2, Stream& stream);
        template <size_t MAX_DISPARITY>
        void winnerTakesAll(const GpuMat& src, GpuMat& left, GpuMat& right, float uniqueness, bool subpixel, cv::cuda::Stream& stream);
    }
}}}

namespace
{
    struct StereoSGMParams
    {
        int numDisparities;
        int P1;
        int P2;
        int uniquenessRatio;
        StereoSGMParams(int numDisparities = 128, int P1 = 10, int P2 = 120, int uniquenessRatio = 5) : numDisparities(numDisparities), P1(P1), P2(P2), uniquenessRatio(uniquenessRatio) {}
    };

    class StereoSGMImpl CV_FINAL: public cuda::StereoSGM
    {
        static constexpr unsigned int NUM_PATHS = 8u;
    public:
        StereoSGMImpl(int numDisparities, int P1, int P2, int uniquenessRatio);

        virtual void compute(InputArray left, InputArray right, OutputArray disparity) CV_OVERRIDE;
        virtual void compute(InputArray left, InputArray right, OutputArray disparity, Stream& stream) CV_OVERRIDE;
        virtual void compute(InputArray left, InputArray right, OutputArray left_disp, OutputArray right_disp, Stream& stream) CV_OVERRIDE;

        virtual int getBlockSize() const CV_OVERRIDE { return 1; }
        virtual void setBlockSize(int /*blockSize*/) CV_OVERRIDE {}

        virtual int getDisp12MaxDiff() const CV_OVERRIDE { return -1; }
        virtual void setDisp12MaxDiff(int /*disp12MaxDiff*/) CV_OVERRIDE {}

        virtual int getMinDisparity() const CV_OVERRIDE { return 1; }
        virtual void setMinDisparity(int /*minDisparity*/) CV_OVERRIDE {}

        virtual int getNumDisparities() const CV_OVERRIDE { return params.numDisparities; }
        virtual void setNumDisparities(int numDisparities) CV_OVERRIDE { params.numDisparities = numDisparities; }

        virtual int getSpeckleWindowSize() const CV_OVERRIDE { return 0; }
        virtual void setSpeckleWindowSize(int /*speckleWindowSize*/) CV_OVERRIDE {}

        virtual int getSpeckleRange() const CV_OVERRIDE { return 0; }
        virtual void setSpeckleRange(int /*speckleRange*/) CV_OVERRIDE {}

        virtual int getP1() const CV_OVERRIDE { return params.P1; }
        virtual void setP1(int P1) CV_OVERRIDE { params.P1 = P1; }

        virtual int getP2() const CV_OVERRIDE { return params.P2; }
        virtual void setP2(int P2) CV_OVERRIDE { params.P2 = P2; }

        virtual int getUniquenessRatio() const CV_OVERRIDE { return params.uniquenessRatio; }
        virtual void setUniquenessRatio(int uniquenessRatio) CV_OVERRIDE { params.uniquenessRatio = uniquenessRatio; }

        virtual int getMode() const CV_OVERRIDE { return -1; }
        virtual void setMode(int /*mode*/) CV_OVERRIDE {}

        virtual int getPreFilterCap() const CV_OVERRIDE { return -1; }
        virtual void setPreFilterCap(int /*preFilterCap*/) CV_OVERRIDE {}

    private:
        StereoSGMParams params;
    };

    StereoSGMImpl::StereoSGMImpl(int numDisparities, int P1, int P2, int uniquenessRatio)
        : params(numDisparities, P1, P2, uniquenessRatio)
    {
    }

    void StereoSGMImpl::compute(InputArray left, InputArray right, OutputArray disparity)
    {
        compute(left, right, disparity, Stream::Null());
    }

    void StereoSGMImpl::compute(InputArray left, InputArray right, OutputArray disparity, Stream& stream)
    {
        compute(left, right, disparity, right_disp_dummy, stream);
    }

    void StereoSGMImpl::compute(InputArray _left, InputArray _right, OutputArray _left_disp, OutputArray _right_disp, Stream& _stream)
    {
        using namespace ::cv::cuda::device::stereosgm;

        GpuMat left = _left.getGpuMat();
        GpuMat right = _right.getGpuMat();
        const Size size = left.size();

        CV_Assert(left.type() == CV_8UC1 || left.type() == CV_16UC1);
        CV_Assert(size == right.size() && left.type() == right.type());

        _left_disp.create(size, CV_16UC1);
        _right_disp.create(size, CV_16UC1);
        GpuMat left_disp = _left_disp.getGpuMat();
        GpuMat right_disp = _right_disp.getGpuMat();

        GpuMat censusedLeft, censusedRight;
        censusedLeft.create(size, CV_32SC1);
        censusedRight.create(size, CV_32SC1);
        censusTransform(left, censusedLeft, _stream);
        censusTransform(right, censusedRight, _stream);

        GpuMat aggregated;
        GpuMat disparityRight;
        aggregated.create(Size(size.width * size.height * params.numDisparities * NUM_PATHS, 1), CV_8UC1);
        disparityRight.create(size, CV_16UC1);

        switch (params.numDisparities)
        {
        case 64:
            pathAggregation<64, NUM_PATHS>(censusedLeft, censusedRight, aggregated, params.P1, params.P2, _stream);
            winnerTakesAll<64>(aggregated, left_disp, right_disp, (float)(100 - params.uniquenessRatio) / 100, true, _stream);
            break;
        case 128:
            pathAggregation<128, NUM_PATHS>(censusedLeft, censusedRight, aggregated, params.P1, params.P2, _stream);
            winnerTakesAll<128>(aggregated, left_disp, right_disp, (float)(100 - params.uniquenessRatio) / 100, true, _stream);
            break;
        default:
            // TODO throw CV Exception
            break;
        }
    }
}

Ptr<cuda::StereoSGM> cv::cuda::createStereoSGM(int numDisparities, int P1, int P2, int uniquenessRatio)
{
    return makePtr<StereoSGMImpl>(numDisparities, P1, P2, uniquenessRatio);
}

#endif /* !defined (HAVE_CUDA) */
