#include "video_extract.h"
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <iostream>
#include <opencv2/videoio/registry.hpp>

void videoExtract::extract_frames(const std::string &videoFilePath , std:: vector< cv::Mat> & frames)
{
	
	try{
	//open the video file
	
	for (auto api : cv::videoio_registry::getBackends()){

		std::cout << cv::videoio_registry::getBackendName(api) << std::endl;
	}
	
  	cv::VideoCapture cap(videoFilePath,cv::CAP_V4L2);
	// std::cout << cap.getBackendName() << std::endl;
	// cap.open(videoFilePath);


  	if(!cap.isOpened())  // check if we succeeded
         std::cout <<"error message: " << "Can not open Video file" << std::endl;
	
  	//cap.get(CV_CAP_PROP_FRAME_COUNT) contains the number of frames in the video;
  	for(int frameNum = 0; frameNum < cap.get(CV_CAP_PROP_FRAME_COUNT);frameNum++)
  	{
  		cv::Mat frame;
  		cap >> frame; // get the next frame from video
  		frames.push_back(frame);
	    std::cout << "good message: " << +frameNum << std::endl;
  	}
	
  }
  catch( cv::Exception& e ){
    std::cout << "error message exception: " << e.msg << std::endl;
    exit(1);
  }
	
}