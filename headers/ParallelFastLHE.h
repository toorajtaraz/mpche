#ifndef PARALLELFASTLHE_H_
#define PARALLELFASTLHE_H_
#include <opencv2/opencv.hpp>
class ParallelFastLHE
{
private:
    int *ExtractHistogram(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt = NULL, int sw = 1);
    int *ExtractHistogramRGB(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt = NULL, int sw = 1);
    double *CalculateProbability(int *hist, int total_pixels);
    double *BuildLookUpTable(double *prob);
    double *BuildLookUpTableRGB(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw = true);
    void BuildAllLuts(std::map<std::tuple<int, int>, double *> &all_luts, cv::Mat img, int offset, int i_start, int i_end, int j_start, int j_end);

public:
    void Test(cv::Mat img);
    void ApplyLHEWithInterpolHelper(cv::Mat &base, cv::Mat img, int window, int i_start, int i_end, std::map<std::tuple<int, int>, double *> all_luts);
    void ApplyLHEWithInterpolation(cv::Mat &base, cv::Mat img, int window);
};

#endif // PARALLELFASTLHE_H_