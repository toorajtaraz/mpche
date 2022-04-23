#ifndef PARALLELLHE_H_
#define PARALLELLHE_H_
#include <opencv2/opencv.hpp>
class ParallelLHE
{
private:
    void ApplyLHEHelper(cv::Mat &base, cv::Mat img, int window, int i_start, int i_end);

public:
    void Test(cv::Mat img);
    void ApplyLHE(cv::Mat &base, cv::Mat img, int window);
};

#endif // PARALLELLHE_H_