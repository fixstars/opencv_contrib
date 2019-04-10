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

#ifdef HAVE_CUDA

namespace opencv_test { namespace {

//////////////////////////////////////////////////////////////////////////
// StereoBM

struct StereoBM : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(StereoBM, Regression)
{
    cv::Mat left_image  = readImage("stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right_image = readImage("stereobm/aloe-R.png", cv::IMREAD_GRAYSCALE);
    cv::Mat disp_gold   = readImage("stereobm/aloe-disp.png", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());

    cv::Ptr<cv::StereoBM> bm = cv::cuda::createStereoBM(128, 19);
    cv::cuda::GpuMat disp;

    bm->compute(loadMat(left_image), loadMat(right_image), disp);

    EXPECT_MAT_NEAR(disp_gold, disp, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Stereo, StereoBM, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////////
// StereoBeliefPropagation

struct StereoBeliefPropagation : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(StereoBeliefPropagation, Regression)
{
    cv::Mat left_image  = readImage("stereobp/aloe-L.png");
    cv::Mat right_image = readImage("stereobp/aloe-R.png");
    cv::Mat disp_gold   = readImage("stereobp/aloe-disp.png", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());

    cv::Ptr<cv::cuda::StereoBeliefPropagation> bp = cv::cuda::createStereoBeliefPropagation(64, 8, 2, CV_16S);
    bp->setMaxDataTerm(25.0);
    bp->setDataWeight(0.1);
    bp->setMaxDiscTerm(15.0);
    bp->setDiscSingleJump(1.0);

    cv::cuda::GpuMat disp;

    bp->compute(loadMat(left_image), loadMat(right_image), disp);

    cv::Mat h_disp(disp);
    h_disp.convertTo(h_disp, disp_gold.depth());

    EXPECT_MAT_NEAR(disp_gold, h_disp, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Stereo, StereoBeliefPropagation, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////////
// StereoConstantSpaceBP

struct StereoConstantSpaceBP : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(StereoConstantSpaceBP, Regression)
{
    cv::Mat left_image  = readImage("csstereobp/aloe-L.png");
    cv::Mat right_image = readImage("csstereobp/aloe-R.png");

    cv::Mat disp_gold;

    if (supportFeature(devInfo, cv::cuda::FEATURE_SET_COMPUTE_20))
        disp_gold = readImage("csstereobp/aloe-disp.png", cv::IMREAD_GRAYSCALE);
    else
        disp_gold = readImage("csstereobp/aloe-disp_CC1X.png", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());

    cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp = cv::cuda::createStereoConstantSpaceBP(128, 16, 4, 4);
    cv::cuda::GpuMat disp;

    csbp->compute(loadMat(left_image), loadMat(right_image), disp);

    cv::Mat h_disp(disp);
    h_disp.convertTo(h_disp, disp_gold.depth());

    EXPECT_MAT_SIMILAR(disp_gold, h_disp, 1e-4);
}

INSTANTIATE_TEST_CASE_P(CUDA_Stereo, StereoConstantSpaceBP, ALL_DEVICES);

////////////////////////////////////////////////////////////////////////////////
// reprojectImageTo3D

PARAM_TEST_CASE(ReprojectImageTo3D, cv::cuda::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(ReprojectImageTo3D, Accuracy)
{
    cv::Mat disp = randomMat(size, depth, 5.0, 30.0);
    cv::Mat Q = randomMat(cv::Size(4, 4), CV_32FC1, 0.1, 1.0);

    cv::cuda::GpuMat dst;
    cv::cuda::reprojectImageTo3D(loadMat(disp, useRoi), dst, Q, 3);

    cv::Mat dst_gold;
    cv::reprojectImageTo3D(disp, dst_gold, Q, false);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(CUDA_Stereo, ReprojectImageTo3D, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16S)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// StereoSGM

struct StereoSGM_Test_Params
{
    String leftSrcImagePath;
    String rightSrcImagePath;
    String leftFixedImagePath;
    int dispScaleFactor;
    int dispUnknVal;
};

std::ostream& operator<<(std::ostream& os, const StereoSGM_Test_Params& params)
{
    os << "(" << params.leftSrcImagePath << ", " << params.rightSrcImagePath << ") - > " << params.leftFixedImagePath;
    os << ", scale factor = " << params.dispScaleFactor << ", unknown value = " << params.dispUnknVal;
    return os;
}

PARAM_TEST_CASE(StereoSGM, cv::cuda::DeviceInfo, StereoSGM_Test_Params)
{
    cv::cuda::DeviceInfo devInfo;
    StereoSGM_Test_Params params;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        params = GET_PARAM(1);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(StereoSGM, Regression)
{
    Mat h_leftImage = imread(params.leftSrcImagePath, IMREAD_GRAYSCALE);
    Mat h_rightImage = imread(params.rightSrcImagePath, IMREAD_GRAYSCALE);
    Mat trueLeftDisp = imread(params.leftFixedImagePath, IMREAD_GRAYSCALE);
    assert(!h_leftImage.empty() && !h_rightImage.empty() && !trueLeftDisp.empty());

    Mat tmp;
    trueLeftDisp.convertTo(tmp, CV_32FC1, 1.f / params.dispScaleFactor);
    trueLeftDisp = tmp;
    tmp.release();

    GpuMat d_leftImage, d_rightImage, d_leftDisp;
    d_leftImage.upload(h_leftImage);
    d_rightImage.upload(h_rightImage);

    cv::Ptr<cv::cuda::StereoSGM> sgm = cv::cuda::createStereoSGM(64, 10, 120, 5);
    sgm->compute(d_leftImage, d_rightImage, d_leftDisp);

    Mat h_leftDisp;
    d_leftDisp.download(h_leftDisp);
    h_leftDisp.convertTo(tmp, CV_32FC1, 1.f / StereoMatcher::DISP_SCALE);
    h_leftDisp = tmp;
    tmp.release();

    Mat unknMask;
    absdiff(trueLeftDisp, Scalar(params.dispUnknVal), unknMask);
    unknMask = unknMask < std::numeric_limits<float>::epsilon();

    double allRMS = 1.0 / sqrt(trueLeftDisp.cols * trueLeftDisp.rows) * cvtest::norm(h_leftDisp, trueLeftDisp, NORM_L2, unknMask);
    EXPECT_LT(allRMS, 5);
}

::std::vector<StereoSGM_Test_Params> generateDatasets4StereoMatching()
{
    const string DATASETS_DIR = "stereomatching/datasets/";
    const string DATASETS_FILE = "datasets.xml";
    const string LEFT_IMG_NAME = "im2.png";
    const string RIGHT_IMG_NAME = "im6.png";
    const string TRUE_LEFT_DISP_NAME = "disp2.png";

    addDataSearchSubDirectory("cv");
    const string dataPath = findDataDirectory(DATASETS_DIR);
    FileStorage datasetsFS(dataPath + DATASETS_FILE, FileStorage::READ);
    assert(datasetsFS.isOpened());

    ::std::vector<StereoSGM_Test_Params> datasets;

    FileNode fn = datasetsFS.getFirstTopLevelNode();
    assert(fn.isSeq());
    for (int i = 0; i < (int)fn.size(); i += 3)
    {
        String _name = fn[i];
        String _dispScaleFactor = fn[i + 1];
        String _dispUnknVal = fn[i + 2];
        datasets.push_back({
            dataPath + _name + "/" + LEFT_IMG_NAME,
            dataPath + _name + "/" + RIGHT_IMG_NAME,
            dataPath + _name + "/" + TRUE_LEFT_DISP_NAME,
            atoi(_dispScaleFactor.c_str()),
            atoi(_dispUnknVal.c_str())
        });
    }

    return datasets;
}

::std::vector<StereoSGM_Test_Params> datasets4StereoMatching = generateDatasets4StereoMatching();

INSTANTIATE_TEST_CASE_P(CUDA_Stereo, StereoSGM, testing::Combine(
    ALL_DEVICES,
    ValuesIn(datasets4StereoMatching.begin(), datasets4StereoMatching.end())
));

}} // namespace
#endif // HAVE_CUDA
