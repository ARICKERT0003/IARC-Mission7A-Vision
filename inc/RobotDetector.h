#ifndef ROBOTDETECTOR
#define ROBOTDETECTOR

#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "Filter.h"

class RobotDetector 
{
  public:
  RobotDetector();
  void load(const std::string&);
  void init();
  void reset();
  void detect(cv::Mat&);
  void getMaskVect(std::vector< cv::Mat>&);
  void getKeyPoints(std::vector< cv::KeyPoint>&);

  private:
  // Mask Size
  int _frameWidth;
  int _frameHeight;

  // Thresholds
  Threshold _redThresh;
  Threshold _greenThresh;
  Threshold _whiteThresh;

  // Frame converted to HSV 
  cv::Mat _hsv;          

  // Masks
  cv::Mat _redMask;          
  cv::Mat _greenMask;        
  cv::Mat _redMaskDilate;    
  cv::Mat _greenMaskDilate;  
  cv::Mat _whiteMask;        
  cv::Mat _redAndWhiteMask;  
  cv::Mat _greenAndWhiteMask;

  // BandFilter
  BandPass _bandpass;
  BandStop _bandstop;

  // Kernels
  int _dilateKernelSize;
  cv::Mat _dilateKernel;

  // Blob Detector
  cv::SimpleBlobDetector::Params _blobParams;
  std::unique_ptr< cv::SimpleBlobDetector> _ptrBlobDetector;
  std::vector< cv::KeyPoint > _redKeyPointVect;
  std::vector< cv::KeyPoint > _greenKeyPointVect;
};
#endif
