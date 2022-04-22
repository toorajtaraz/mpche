#ifndef PARALLELLHE_H_
#define PARALLELLHE_H_
#include <opencv2/opencv.hpp>
class ParallelLHE
{
private:
    int *ExtractHistogram(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt = NULL, int sw = 1);
    int *ExtractHistogramRGB(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt = NULL, int sw = 1);
    double *CalculateProbability(int *hist, int total_pixels);
    double *BuildLookUpTable(double *prob);
    double *BuildLookUpTableRGB(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw = true);
    void ApplyLHEHelper(cv::Mat &base, cv::Mat img, int window, int i_start, int i_end);

public:
    void Test(cv::Mat img);
    void ApplyLHE(cv::Mat &base, cv::Mat img, int window);
};

#endif // PARALLELLHE_H_