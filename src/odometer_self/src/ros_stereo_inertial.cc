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
// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/calib3d/calib3d.hpp>
// #include <opencv2/core/core.hpp>
// #include <Eigen/Core>
// #include <Eigen/Geometry>
// #include <Eigen/Dense>
// #include "opencv2/core/eigen.hpp"

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>

#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/Imu.h>

#include <geometry_msgs/PoseStamped.h> 
#include <tf/tf.h> 
#include <tf/transform_datatypes.h> 
#include"Converter.h"


#include<opencv2/core/core.hpp>

#include"./include/System.h"
#include"ImuTypes.h"
#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};
class ImageGrabber
{
private:
		// ros::NodeHandle n; 
		// //定义订阅者
		// ros::Subscriber sub_imu;
		// ros::Subscriber sub_img_left;
		// ros::Subscriber sub_img_right;
		// //定义发布者与订阅者
		// ros::Publisher pose_pub;
    // geometry_msgs::PointStamped pose;
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bRect, const bool bClahe): mpSLAM(pSLAM), mpImuGb(pImuGb), do_rectify(bRect), mbClahe(bClahe){}

    void GrabImageLeft(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageRight(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    cv::Mat  SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft,mBufMutexRight;
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool do_rectify;
    cv::Mat M1l,M2l,M1r,M2r;
    ros::Publisher pose_pub;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "Stereo_Inertial");

std::cout<<"**********************************************"<<std::endl;

  bool bEqual = false;
  if(argc < 4 || argc > 5)
  {
    cerr << endl << "Usage: rosrun ORB_SLAM3 Stereo_Inertial path_to_vocabulary path_to_settings do_rectify [do_equalize]" << endl;
    ros::shutdown();
    return 1;
  }

  std::string sbRect(argv[3]);
  if(argc==5)
  {
    std::string sbEqual(argv[4]);
    if(sbEqual == "true")
      bEqual = true;
  }

//   // Create SLAM system. It initializes all system threads and gets ready to process frames.
//   //初始化SLAM系统
  ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO,true);


  ImuGrabber imugb;
  ImageGrabber igb(&SLAM,&imugb,sbRect == "true",bEqual);
  // 去畸变
    if(igb.do_rectify)
    {      
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
                rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,igb.M1l,igb.M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,igb.M1r,igb.M2r);
    }

  // Maximum delay, 5 seconds
  //抓取数据

  ros::NodeHandle n("~");
  ros::Publisher pose_pub = n.advertise<geometry_msgs::PoseStamped>("orb_pose", 100);//modify
  igb.pose_pub=pose_pub;
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  ros::Subscriber sub_imu = n.subscribe("/imu", 1000, &ImuGrabber::GrabImu, &imugb); 
  ros::Subscriber sub_img_left = n.subscribe("/camera/left/image_raw", 100, &ImageGrabber::GrabImageLeft,&igb);
  ros::Subscriber sub_img_right = n.subscribe("/camera/right/image_raw", 100, &ImageGrabber::GrabImageRight,&igb);
  //新开线程
  std::thread sync_thread(&ImageGrabber::SyncWithImu,&igb);
  ros::spin();
  return 0;
}



//抓取左目数据，打开线程，加入数据，关闭线程，在imgLeftBuf中
void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg)
{
  mBufMutexLeft.lock();
  if (!imgLeftBuf.empty())
    imgLeftBuf.pop();
  imgLeftBuf.push(img_msg);
  mBufMutexLeft.unlock();
}
//抓取右目数据，打开线程，加入数据，关闭线程
void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg)
{
  mBufMutexRight.lock();
  if (!imgRightBuf.empty())
    imgRightBuf.pop();
  imgRightBuf.push(img_msg);
  mBufMutexRight.unlock();
}
//把sensor_msgs格式改成cv::mat格式
cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  
  if(cv_ptr->image.type()==0)
  {
    return cv_ptr->image.clone();
  }
  else
  {
    std::cout << "Error type" << std::endl;
    return cv_ptr->image.clone();
  }
}

cv::Mat  ImageGrabber::SyncWithImu()
{
  const double maxTimeDiff = 0.01;
  while(1)
  {
    cv::Mat imLeft, imRight;
    double tImLeft = 0, tImRight = 0;
    //左右目和IMU均不为空的情况下
    if (!imgLeftBuf.empty()&&!imgRightBuf.empty()&&!mpImuGb->imuBuf.empty())
    {
      tImLeft = imgLeftBuf.front()->header.stamp.toSec();
      tImRight = imgRightBuf.front()->header.stamp.toSec();

      this->mBufMutexRight.lock();
      //左目比右目时间长，且右目中不止一帧，取最新的一个或者取时间阈值满足要求的一个
      while((tImLeft-tImRight)>maxTimeDiff && imgRightBuf.size()>1)
      {
        imgRightBuf.pop();
        tImRight = imgRightBuf.front()->header.stamp.toSec();
      }
      this->mBufMutexRight.unlock();

      this->mBufMutexLeft.lock();
      //右目比左目时间长，且右目中不止一帧，取最新的一个或者取时间阈值满足要求的一个
      while((tImRight-tImLeft)>maxTimeDiff && imgLeftBuf.size()>1)
      {
        imgLeftBuf.pop();
        tImLeft = imgLeftBuf.front()->header.stamp.toSec();
      }
      this->mBufMutexLeft.unlock();

      if((tImLeft-tImRight)>maxTimeDiff || (tImRight-tImLeft)>maxTimeDiff)
      {
        // std::cout << "big time difference" << std::endl;
        continue;
      }
      if(tImLeft>mpImuGb->imuBuf.back()->header.stamp.toSec())
          continue;
      // 获取cv格式下的左右目图片
      this->mBufMutexLeft.lock();
      imLeft = GetImage(imgLeftBuf.front());
      imgLeftBuf.pop();
      this->mBufMutexLeft.unlock();

      this->mBufMutexRight.lock();
      imRight = GetImage(imgRightBuf.front());
      imgRightBuf.pop();
      this->mBufMutexRight.unlock();

      vector<ORB_SLAM3::IMU::Point> vImuMeas;
      mpImuGb->mBufMutex.lock();
      if(!mpImuGb->imuBuf.empty())
      {
        // Load imu measurements from buffer
        vImuMeas.clear();
        //imuBuf不为空并且并且其时间戳小于左目
        //读取IMU数据
        while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tImLeft)
        {
          double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
          cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
          cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
          vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
          mpImuGb->imuBuf.pop();
        }
      }
      mpImuGb->mBufMutex.unlock();
      //直方图均衡化
      if(mbClahe)
      {
        mClahe->apply(imLeft,imLeft);
        mClahe->apply(imRight,imRight);
      }
      //去畸变
      if(do_rectify)
      {
        cv::remap(imLeft,imLeft,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRight,M1r,M2r,cv::INTER_LINEAR);
      }
      cv::Mat Tcw;

      Tcw=mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);
      // std::cout<<Tcw.size().height<<std::endl;
      // std::cout<<Tcw.size().width<<std::endl;
      // std::cout<<Tcw.channels()<<std::endl;
if(Tcw.size().height!=0){
      geometry_msgs::PoseStamped pose;
      pose.header.stamp = ros::Time::now();
      pose.header.frame_id ="map";
      cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t(); // Rotation information
      cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3); // translation information
      vector<float> q = ORB_SLAM3::Converter::toQuaternion(Rwc);

      tf::Transform new_transform;
      new_transform.setOrigin(tf::Vector3(twc.at<float>(0, 0), twc.at<float>(0, 1), twc.at<float>(0, 2)));

      tf::Quaternion quaternion(q[0], q[1], q[2], q[3]);
      new_transform.setRotation(quaternion);

      tf::poseTFToMsg(new_transform, pose.pose);

      pose_pub.publish(pose);
}
      std::chrono::milliseconds tSleep(1);
      std::this_thread::sleep_for(tSleep);
    }
  }
}
//读取imu数据并存储进去
void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}


/*
左目：
      this->mBufMutexLeft.lock();
      imLeft = GetImage(imgLeftBuf.front());
      imgLeftBuf.pop();
      this->mBufMutexLeft.unlock();
*/

// cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<ORB_SLAM3::IMU::Point>& vImuMeas, string filename)
// {
//   cv::Mat Tcw =GrabImageStereo(imLeft,imRight,timestamp,filename);
// }

// cv::Mat GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
// {
//     cv::Mat mImGray = imRectLeft;
//     cv::Mat imGrayRight = imRectRight;
//     cv::Mat mImRight = imRectRight;
//     // step 1 ：将RGB或RGBA图像转为灰度图像
//     if(mImGray.channels()==3)
//     {
//       cvtColor(mImGray,mImGray,CV_RGB2GRAY);
//       cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
//     }
//     int nFeatures, nLevels, fIniThFAST, fMinThFAST;
//     float fScaleFactor;
//     ORBextractor*mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

//     ORB_SLAM3::Frame mCurrentFrame= ORB_SLAM3::Frame(
//             mImGray,                // 左目图像
//             imGrayRight,            // 右目图像
//             timestamp,              // 时间戳
//             mpORBextractorLeft,     // 左目特征提取器
//             mpORBextractorRight,    // 右目特征提取器
//             mpORBVocabulary,        // 字典
//             mK,                     // 内参矩阵
//             mDistCoef,              // 去畸变参数
//             mbf,                    // 基线长度
//             mThDepth,				// 远点,近点的区分阈值
//             mpCamera);				// 相机模型

// }
