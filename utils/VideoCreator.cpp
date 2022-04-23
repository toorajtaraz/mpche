#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "VideoCreator.h"
using namespace std;

void VideoCreator::videoHandlerPipeline(char *input_path, char *output_path, int color, int mode, int thread_num, int w)
{
    try
    {
        cv::VideoCapture cap(input_path, cv::CAP_FFMPEG);
        // std::cout << cap.getBackendName() << std::endl;
        // cap.open(videoFilePath);

        if (!cap.isOpened())
        {
            std::cout << "Error: "
                      << "Can not open Video file at " << input_path << std::endl;
        }

        for (int frame_num = 0; frame_num < cap.get(cv::CAP_PROP_FRAME_COUNT); frame_num++)
        {
            cv::Mat img;
            cap >> img; // get the next frame from video

            std::cout << "good message: " << +frame_num << std::endl;
        }
    }
    catch (cv::Exception &e)
    {
        std::cout << "error message exception: " << e.msg << std::endl;
        exit(1);
    }
}