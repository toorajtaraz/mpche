#ifndef SERIALLHE_H_
#define SERIALLHE_H_
#include <opencv2/opencv.hpp>
class SerialLHE
{

public:
    void Test(cv::Mat img);
    void ApplyLHEWithInterpol(cv::Mat &base, cv::Mat img, int window);
    void ApplyLHE(cv::Mat &base, cv::Mat img, int window);
};

#endif // SERIALLHE_H_