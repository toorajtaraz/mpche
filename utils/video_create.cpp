#include <iostream> 
#include <string>   
#include <vector>
#include <opencv2/core/core.hpp>        
#include <opencv2/highgui/highgui.hpp>  
#include "video_create.h"
using namespace std;
using namespace cv;

void VideoCreator::create_video(const std::string &videoFilePath , std:: vector<cv::Mat> & images){

    Size size = images[0].size(); 
    cout  << size << endl;
    VideoWriter output(videoFilePath, VideoWriter::fourcc('M', 'J', 'P', 'G'), 1, size);

    if (!output.isOpened()){
        cout  << "Could not open the output video for write: "<< endl;
    }

    for(auto res : images){
         output.write(res);
    }
    output.release();
   
}