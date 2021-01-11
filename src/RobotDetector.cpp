#include "RobotDetector.h"

void RobotDetector::load(const std::string& file)
{
  YAML::Node RobotDoc = YAML::LoadFile(file.c_str());

  const YAML::Node frame = RobotDoc["Frame"];
  _frameWidth = frame[ "width" ].as<int>();
  _frameHeight = frame[ "height" ].as<int>();

  const YAML::Node kernel = RobotDoc["dilateRedGreen"];
  _dilateKernelSize = kernel[ "size" ].as<int>();

  _redThresh.load(file, "red_hsv");
  _greenThresh.load(file, "green_hsv");
  _whiteThresh.load(file, "white_hsv");
}

void RobotDetector::init()
{
  // Set mask size and types
  _redMask           = cv::Mat(_frameHeight, _frameWidth, CV_8UC1); 
  _greenMask         = cv::Mat(_frameHeight, _frameWidth, CV_8UC1); 
  _redMaskDilate     = cv::Mat(_frameHeight, _frameWidth, CV_8UC1); 
  _greenMaskDilate   = cv::Mat(_frameHeight, _frameWidth, CV_8UC1); 
  _whiteMask         = cv::Mat(_frameHeight, _frameWidth, CV_8UC1); 
  _redAndWhiteMask   = cv::Mat(_frameHeight, _frameWidth, CV_8UC1); 
  _greenAndWhiteMask = cv::Mat(_frameHeight, _frameWidth, CV_8UC1); 
  
  // Create Dilation Kernels
  _dilateKernel = cv::Mat::ones(_dilateKernelSize, _dilateKernelSize, CV_8UC1);

  // Set up Blob Detector
  _blobParams.blobColor = (uint8_t)1;
  _blobParams.filterByArea = true;
  _blobParams.filterByCircularity = false;
  _blobParams.filterByColor = false;
  _blobParams.filterByConvexity = false;
  _blobParams.filterByInertia = false;

  _blobParams.maxArea = 100;
  //_ptrBlobDetector = std::make_unique< cv::SimpleBlobDetector >(_blobParams);
}

void RobotDetector::reset()
{
  // Set values of mats to 0
  _redMask.setTo(0);
  _greenMask.setTo(0);
  _redMaskDilate.setTo(0);
  _greenMaskDilate.setTo(0);
  _whiteMask.setTo(0);
  _redAndWhiteMask.setTo(0);
  _greenAndWhiteMask.setTo(0);

  // Clear KeyPoints
  _redKeyPointVect.clear();
  _greenKeyPointVect.clear();
}

void RobotDetector::detect(cv::Mat& frame) 
{
  cv::Mat mask_C0, mask_C1, mask_C2;
  std::array<cv::Mat, 3> hsvArray;

  // Change Color Space
  cv::cvtColor(frame, _hsv, cv::COLOR_BGR2HSV);
  
  // Split into channels
  cv::split(_hsv, hsvArray.data());

  // Switch Mask (Red/Green)
  _bandstop.getMask(hsvArray[0], _redThresh.threshVect[0], _redThresh.threshVect[3], mask_C0);
  _bandpass.getMask(hsvArray[1], _redThresh.threshVect[1], _redThresh.threshVect[4], mask_C1);
  _bandpass.getMask(hsvArray[2], _redThresh.threshVect[2], _redThresh.threshVect[5], mask_C2);
  cv::bitwise_and(mask_C0, mask_C1, _redMask, mask_C2);

  // Grow Region
  cv::dilate(_redMask, _redMaskDilate, _dilateKernel);
  cv::dilate(_greenMask, _greenMaskDilate, _dilateKernel);

  // White Mask
  _bandpass.getMask(frame, _whiteThresh, _whiteMask); 
  
  // And switch with white regions
  cv::bitwise_and(_redMask, _whiteMask, _redAndWhiteMask);
  cv::bitwise_and(_greenMask, _whiteMask, _greenAndWhiteMask);

  // Detect Blobs
  //_ptrBlobDetector->detect(_redAndWhiteMask, _redKeyPointVect); 
  //_ptrBlobDetector->detect(_greenAndWhiteMask, _greenKeyPointVect); 
}

void RobotDetector::getMaskVect(std::vector< cv::Mat>& maskVect)
{
  // Clear vector
  maskVect.clear();

  // COPY masks into vect
  maskVect.push_back(_redMask);
  maskVect.push_back(_greenMask);
  maskVect.push_back(_redMaskDilate);
  maskVect.push_back(_greenMaskDilate);
  maskVect.push_back(_whiteMask);
  maskVect.push_back(_redAndWhiteMask);
  maskVect.push_back(_greenAndWhiteMask);
}

void RobotDetector::getKeyPoints(std::vector< cv::KeyPoint>& keyPointVect)
{
  // Clear vector
  keyPointVect.clear();

  // COPY keypoints into vector
  for(int i=0; i<_redKeyPointVect.size(); i++)
  {
    keyPointVect.push_back(_redKeyPointVect[i]);
  }

  for(int i=0; i<_greenKeyPointVect.size(); i++)
  {
    keyPointVect.push_back(_greenKeyPointVect[i]);
  }
}

