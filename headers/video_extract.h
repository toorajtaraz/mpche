#ifndef VIDEO_EXTRACT_H_
#define VIDEO_EXTRACT_H_
#include <opencv2/opencv.hpp>

class videoExtract
{

public:
    void extract_frames(const std::string &videoFilePath , std:: vector< cv::Mat> & frames);
};

#endif // VIDEO_EXTRACT_H_