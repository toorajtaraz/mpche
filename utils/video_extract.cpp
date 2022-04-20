#include "headers/video_extract.h"
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <iostream>


void videoExtract::extract_frames(const std::string &videoFilePath , std:: vector< cv::Mat> & frames)
{
	
	try{
	//open the video file
  	cv::VideoCapture cap(videoFilePath);
  	if(!cap.isOpened())  // check if we succeeded
        std::cout <<"error message: " << "Can not open Video file" << std::endl;
	
  	//cap.get(CV_CAP_PROP_FRAME_COUNT) contains the number of frames in the video;
  	for(int frameNum = 0; frameNum < cap.get(CV_CAP_PROP_FRAME_COUNT);frameNum++)
  	{
  		cv::Mat frame;
  		cap >> frame; // get the next frame from video
  		frames.push_back(frame);
  	}
  }
  catch( cv::Exception& e ){
    std::cout << "error message: " << e.msg << std::endl;
    exit(1);
  }
	
}