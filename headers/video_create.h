#ifndef VIDEO_CREATE_H_
#define VIDEO_CREATE_H_
#include <opencv2/opencv.hpp>

class VideoCreator
{

public:
    void create_video(const std::string &videoFilePath , std:: vector<cv::Mat> & images);
};

#endif // VIDEO_CREATE_H_