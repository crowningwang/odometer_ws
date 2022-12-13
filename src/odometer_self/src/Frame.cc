/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false)
{
}


//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr.clone()), mRlr(frame.mRlr.clone()), mtlr(frame.mtlr.clone()), mTrl(frame.mTrl.clone()), mTimeStereoMatch(frame.mTimeStereoMatch), mTimeORB_Ext(frame.mTimeORB_Ext)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);

    if(!frame.mVw.empty())
        mVw = frame.mVw.clone();

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;
}

// 立体匹配模式下的双目
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    // Step 1 帧的ID 自增
    mnId=nNextId++;

    // Scale Level Info
    // Step 2 计算图像金字塔的参数 
	// 获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    // 获得层与层之间的缩放比
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    // 计算上面缩放比的对数
    mfLogScaleFactor = log(mfScaleFactor);
    // 获取每层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    // 同样获取每层图像缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    // 高斯模糊的时候，使用的方差
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    // 获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // Step 3 对左目右目图像提取ORB特征点, 第一个参数0-左图， 1-右图。为加速计算，同时开了两个线程计算
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    // 对右目图像提取orb特征
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    // 等待两张图像特征点提取过程完成
    threadLeft.join();
    threadRight.join();
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    // mvKeys中保存的是左图像中的特征点，这里是获取左侧图像中特征点的个数
    N = mvKeys.size();
    // 如果左图像中没有成功提取到特征点那么就返回，也意味这这一帧的图像无法使用
    if(mvKeys.empty())
        return;
    // Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正
    UndistortKeyPoints();

#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    // Step 5 计算双目间特征点的匹配，只有匹配成功的特征点会计算其深度,深度存放在 mvDepth  
    // mvuRight中存储的应该是左图像中的点所匹配的在右图像中的点的横坐标（纵坐标相同）
    ComputeStereoMatches();
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    // 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    //  Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);
        // 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        // 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        // 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;
        // 特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }
    // 双目相机基线长度
    mb = mbf/fx;
    //？？？？？？/这里和原论文不一样
    if(pPrevF)
    {
        if(!pPrevF->mVw.empty())
            mVw = pPrevF->mVw.clone();
    }
    else
    {
        mVw = cv::Mat::zeros(3,1,CV_32F);
    }
    // Step 6 将特征点分配到图像网格中
    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3,4,CV_32F);
    mTrl = cv::Mat(3,4,CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,0);
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3,4,CV_32F);
    mTrl = cv::Mat(3,4,CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,1000);
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3,4,CV_32F);
    mTrl = cv::Mat(3,4,CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    // mVw = cv::Mat::zeros(3,1,CV_32F);
    if(pPrevF)
    {
        if(!pPrevF->mVw.empty())
            mVw = pPrevF->mVw.clone();
    }
    else
    {
        mVw = cv::Mat::zeros(3,1,CV_32F);
    }

    mpMutexImu = new std::mutex();
}


void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    // Step 1  给存储特征点的网格数组 Frame::mGrid 预分配空间
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);
    // 开始对mGrid这个二维数组中的每一个vector元素遍历并预分配空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }

    // Step 2 遍历每个特征点，将每个特征点在mvKeysUn中的索引值放到对应的网格mGrid中

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];
        // 存储某个特征点所在网格的网格坐标，nGridPosX范围：[0,FRAME_GRID_COLS], nGridPosY范围：[0,FRAME_GRID_ROWS]
        int nGridPosX, nGridPosY;
        // 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
            // 如果找到特征点所在网格坐标，将这个特征点的索引添加到对应网格的数组mGrid中
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0)
        monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::GetPose(cv::Mat &Tcw)
{
    Tcw = mTcw.clone();
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(const cv::Mat &Vwb)
{
    mVw = Vwb.clone();
}
/** 
 * @brief 赋值位姿与速度
 */
void Frame::SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb)
{
    mVw = Vwb.clone();
    cv::Mat Rbw = Rwb.t();
    cv::Mat tbw = -Rbw*twb;
    cv::Mat Tbw = cv::Mat::eye(4,4,CV_32F);
    Rbw.copyTo(Tbw.rowRange(0,3).colRange(0,3));
    tbw.copyTo(Tbw.rowRange(0,3).col(3));
    mTcw = mImuCalib.Tcb*Tbw;
    UpdatePoseMatrices();
}


/** 
 * @brief 通过mTcw更新其他关于位姿的变量
 */
void Frame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

cv::Mat Frame::GetImuPosition()
{
    return mRwc*mImuCalib.Tcb.rowRange(0,3).col(3)+mOw;
}

cv::Mat Frame::GetImuRotation()
{
    return mRwc*mImuCalib.Tcb.rowRange(0,3).colRange(0,3);
}

cv::Mat Frame::GetImuPose()
{
    cv::Mat Twb = cv::Mat::eye(4,4,CV_32F);
    Twb.rowRange(0,3).colRange(0,3) = mRwc*mImuCalib.Tcb.rowRange(0,3).colRange(0,3);
    Twb.rowRange(0,3).col(3) = mRwc*mImuCalib.Tcb.rowRange(0,3).col(3)+mOw;
    return Twb.clone();
}


bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        // cout << "\na";
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // cout << "b";

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw*P+mtcw;
        const float Pc_dist = cv::norm(Pc);

        // Check positive depth
        const float &PcZ = Pc.at<float>(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const cv::Point2f uv = mpCamera->project(Pc);

        // cout << "c";

        if(uv.x<mnMinX || uv.x>mnMaxX)
            return false;
        if(uv.y<mnMinY || uv.y>mnMaxY)
            return false;

        // cout << "d";
        pMP->mTrackProjX = uv.x;
        pMP->mTrackProjY = uv.y;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P-mOw;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            return false;

        // cout << "e";

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        // cout << "f";

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // cout << "g";

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv.x;
        pMP->mTrackProjXR = uv.x - mbf*invz;

        pMP->mTrackDepth = Pc_dist;
        // cout << "h";

        pMP->mTrackProjY = uv.y;
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        // cout << "i";

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    // cout << "c";

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

cv::Mat Frame::inRefCoordinates(cv::Mat pCw)
{
    return mRcw*pCw+mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    /*cout << "fX " << factorX << endl;
    cout << "fY " << factorY << endl;*/

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

/**
 * @brief 计算当前帧特征点对应的词袋Bow，主要是mBowVec 和 mFeatVec
 * 
 */
void Frame::ComputeBoW()
{
        // 判断是否以前已经计算过了，计算过了就跳过
    if(mBowVec.empty())
    {
                // 将描述子mDescriptors转换为DBOW要求的输入格式
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,    //当前的描述子vector
                                   mBowVec,         //输出，词袋向量，记录的是单词的id及其对应权重TF-IDF值
                                   mFeatVec,        //输出，记录node id及其对应的图像 feature对应的索引
                                   4);              //4表示从叶节点向前数的层数
    }
}

void Frame::UndistortKeyPoints()
{
    // Step 1 如果第一个畸变参数为0，不需要矫正。第一个畸变参数k1是最重要的，一般不为0，为0的话，说明畸变参数都是0
	// 变量mDistCoef中存储了opencv指定格式的去畸变参数，格式为：(k1,k2,p1,p2,k3)
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    // Step 2 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    // Fill matrix with points
    // N为提取的特征点数量，为满足OpenCV函数输入要求，将N个特征点保存在N*2的矩阵中
    cv::Mat mat(N,2,CV_32F);
    // 遍历每个特征点，并将它们的坐标保存到矩阵中
    for(int i=0; i<N; i++)
    {
        // 然后将这个特征点的横纵坐标分别保存
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // 函数reshape(int cn,int rows=0) 其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    // 为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
    // cv::undistortPoints最后一个矩阵为空矩阵时，得到的点为归一化坐标点
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    // Step 3 存储校正后的特征点
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        // 根据索引获取这个特征点
		// 注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
        cv::KeyPoint kp = mvKeys[i];
        // 读取校正后的坐标并覆盖老坐标
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}
/**
 * @brief 计算去畸变图像的边界
 * 
 * @param[in] imLeft            需要计算边界的图像
 */
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 保存矫正前的图像四个边界点坐标： (0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;
        // 和前面校正特征点一样的操作，将这几个边界点作为输入进行校正
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        // 校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));  // 左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));  // 右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));  // 左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));  // 左下和右下纵坐标最小的
    }
    else
    {
        // 如果畸变参数为0，就直接获得图像边界
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
        /*两帧图像稀疏立体匹配（即：ORB特征点匹配，非逐像素的密集匹配，但依然满足行对齐）
     * 输入：两帧立体矫正后的图像img_left 和 img_right 对应的orb特征点集
     * 过程：
          1. 行特征点统计. 统计img_right每一行上的ORB特征点集，便于使用立体匹配思路(行搜索/极线搜索）进行同名点搜索, 避免逐像素的判断.
          2. 粗匹配. 根据步骤1的结果，对img_left第i行的orb特征点pi，在img_right的第i行上的orb特征点集中搜索相似orb特征点, 得到qi
          3. 精确匹配. 以点qi为中心，半径为r的范围内，进行块匹配（归一化SAD），进一步优化匹配结果
          4. 亚像素精度优化. 步骤3得到的视差为uchar/int类型精度，并不一定是真实视差，通过亚像素差值（抛物线插值)获取float精度的真实视差
          5. 最优视差值/深度选择. 通过胜者为王算法（WTA）获取最佳匹配点。
          6. 删除离缺点(outliers). 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是正确匹配，比如光照变化、弱纹理等会造成误匹配
     * 输出：稀疏特征点视差图/深度图（亚像素精度）mvDepth 匹配结果 mvuRight
     */

    // 为匹配结果预先分配内存，数据类型为float型
    // mvuRight存储右图匹配点索引
    // mvDepth存储特征点的深度信息
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);
    // orb特征相似度阈值  -> mean ～= (max  + min) / 2
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;
    // 金字塔顶层（0层）图像高 nRows
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    // 二维vector存储每一行的orb特征点的列坐标的索引，为什么是vector，因为每一行的特征点有可能不一样，例如
    // vRowIndices[0] = [1，2，5，8, 11]   第1行有5个特征点,他们的列号（即x坐标）分别是1,2,5,8,11
    // vRowIndices[1] = [2，6，7，9, 13, 17, 20]  第2行有7个特征点.etc
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);
    // 右图特征点数量，N表示数量 r表示右图，且不能被修改
    const int Nr = mvKeysRight.size();
    // Step 1. 行特征点统计. 考虑到尺度金字塔特征，一个特征点可能存在于多行，而非唯一的一行
    for(int iR=0; iR<Nr; iR++)
    {
        // 获取特征点ir的y坐标，即行号
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        // 计算特征点ir在行方向上，可能的偏移范围r，即可能的行号为[kpY + r, kpY -r]
        // 2 表示在全尺寸(scale = 1)的情况下，假设有2个像素的偏移，随着尺度变化，r也跟着变化
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);
        // 将特征点ir保证在可能的行号中
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }
    // Step 2 -> 3. 粗匹配 + 精匹配
    // 对于立体矫正后的两张图，在列方向(x)存在最大视差maxd和最小视差mind
    // 也即是左图中任何一点p，在右图上的匹配点的范围为应该是[p - maxd, p - mind], 而不需要遍历每一行所有的像素
    // maxd = baseline * length_focal / minZ
    // mind = baseline * length_focal / maxZ
    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
     // 保存sad块匹配相似度和左图特征点索引
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);
    // 为左图每一个特征点il，在右图搜索最相似的特征点ir
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;
        // 获取左图特征点il所在行，以及在右图对应行中可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;
        // 计算理论上的最佳搜索范围
        const float minU = uL-maxD;
        const float maxU = uL-minD;
        // 最大搜索范围小于0，说明无匹配点
        if(maxU<0)
            continue;
        // 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        // Step2. 粗配准. 左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的相似度和索引
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];
            // 左图特征点il与带匹配点ic的空间尺度差超过2，放弃
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;
            // 使用列坐标(x)进行匹配，和stereomatch一样
            const float &uR = kpR.pt.x;
            // 超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if(uR>=minU && uR<=maxU)
            {
                // 计算匹配点il和待匹配点ic的相似度dist
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);
                // 统计最小相似度及其对应的列坐标(x)
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        // 如果刚才匹配过程中的最佳描述子距离小于给定的阈值
        // Step 3. 精确匹配. 
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            // 计算右图特征点x坐标和对应的金字塔尺度
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            // 尺度缩放后的左右图特征点坐标
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            // 滑动窗口搜索, 类似模版卷积或滤波
            // w表示sad相似度的窗口半径
            const int w = 5;
            // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            //IL.convertTo(IL,CV_32F);
            //IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);
            //???????这个源程序没有？？？？？？
            IL.convertTo(IL,CV_16S);
            IL = IL - IL.at<short>(w,w);
            // 初始化最佳相似度
            int bestDist = INT_MAX;
            // 通过滑动窗口搜索优化，得到的列坐标偏移量
            int bestincR = 0;
            // 滑动窗口的滑动范围为（-L, L）
            const int L = 5;
            // 初始化存储图像块相似度
            vector<float> vDists;
            vDists.resize(2*L+1);
            // 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
            // 列方向起点 iniu = r0 + 最大窗口滑动范围 - 图像块尺寸
            // 列方向终点 eniu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
            // 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            // 判断搜索是否越界
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;
            // 在搜索范围内从左到右滑动，并计算图像块相似度
            for(int incR=-L; incR<=+L; incR++)
            {
                // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                //IR.convertTo(IR,CV_32F);
                //IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);
                // ？？？这个源码也没有？？？？？
                IR.convertTo(IR,CV_16S);
                IR = IR - IR.at<short>(w,w);
                // sad 计算
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                // 统计最小sad和偏移量
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }
                // L+incR 为refine后的匹配点列坐标(x)
                vDists[L+incR] = dist;
            }
            // 搜索窗口越界判断ß 
            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            // Step 4. 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线
            // 使用3点拟合抛物线的方式，用极小值代替之前计算的最优是差值
            //    \                 / <- 由视差为14，15，16的相似度拟合的抛物线
            //      .             .(16)
            //         .14     .(15) <- int/uchar最佳视差值
            //              . 
            //           （14.5）<- 真实的视差值
            //   deltaR = 15.5 - 16 = -0.5
            // 公式参考opencv sgbm源码中的亚像素插值公式
            // 或论文<<On Building an Accurate Stereo Matching System on Graphics Hardware>> 公式7
            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));
            // 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // 根据亚像素精度偏移量delta调整最佳匹配索引
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                // 如果存在负视差，则约束为0.01
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // 根据视差值计算深度信息
                // 保存最相似点的列坐标(x)信息
                // 保存归一化sad最小相似度
                // Step 5. 最优视差值/深度选择.
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }
    // Step 6. 删除离缺点(outliers)
    // 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
    // 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            // 误匹配点置为-1，和初始化时保持一直，作为error code
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}
/** 
 * @brief 当某个特征点的深度信息或者双目信息有效时,将它反投影到三维世界坐标系中
 */
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, cv::Mat& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2), mTlr(Tlr)
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    mRlr = mTlr.rowRange(0,3).colRange(0,3);
    mtlr = mTlr.col(3);

    cv::Mat Rrl = mTlr.rowRange(0,3).colRange(0,3).t();
    cv::Mat trl = Rrl * (-1 * mTlr.col(3));

    cv::hconcat(Rrl,trl,mTrl);

    ComputeStereoFishEyeMatches();
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();
    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    double t_read = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();
    double t_orbextract = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
    double t_stereomatches = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t3 - t2).count();
    double t_assign = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t4 - t3).count();
    double t_undistort = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t5 - t4).count();

    /*cout << "Reading time: " << t_read << endl;
    cout << "Extraction time: " << t_orbextract << endl;
    cout << "Matching time: " << t_stereomatches << endl;
    cout << "Assignment time: " << t_assign << endl;
    cout << "Undistortion time: " << t_undistort << endl;*/

}

void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<cv::Mat>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            cv::Mat p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D.clone();
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    cv::Mat mR, mt, twc;
    if(bRight){
        cv::Mat Rrl = mTrl.colRange(0,3).rowRange(0,3);
        cv::Mat trl = mTrl.col(3);
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.rowRange(0,3).col(3) + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    cv::Mat Pc = mR*P+mt;
    const float Pc_dist = cv::norm(Pc);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    cv::Point2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv.x<mnMinX || uv.x>mnMaxX)
        return false;
    if(uv.y<mnMinY || uv.y>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-twc;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv.x;
        pMP->mTrackProjYR = uv.y;
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv.x;
        pMP->mTrackProjY = uv.y;
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

cv::Mat Frame::UnprojectStereoFishEye(const int &i){
    return mRwc*mvStereo3Dpoints[i]+mOw;
}

} //namespace ORB_SLAM
