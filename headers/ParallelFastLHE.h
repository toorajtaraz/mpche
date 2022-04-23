#ifndef PARALLELFASTLHE_H_
#define PARALLELFASTLHE_H_
#include <opencv2/opencv.hpp>
class ParallelFastLHE
{
private:
    void BuildAllLuts(std::map<std::tuple<int, int>, double *> &all_luts, cv::Mat img, int offset, int i_start, int i_end, int j_start, int j_end);

public:
    void Test(cv::Mat img);
    void ApplyLHEWithInterpolHelper(cv::Mat &base, cv::Mat img, int window, int i_start, int i_end, std::map<std::tuple<int, int>, double *> all_luts);
    void ApplyLHEWithInterpolation(cv::Mat &base, cv::Mat img, int window);
};

#endif // PARALLELFASTLHE_H_