#ifndef SERIALLHE_H_
#define SERIALLHE_H_
#include <opencv2/opencv.hpp>
class SerialLHE
{
private:
    int *ExtractHistogram(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt = NULL, int sw = 1);
    int *ExtractHistogramRGB(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt = NULL, int sw = 1);
    double *CalculateProbability(int *hist, int total_pixels);
    double *BuildLookUpTable(double *prob);
    double *BuildLookUpTableRGB(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw = true);

public:
    void Test(cv::Mat img,std:: vector< cv::Mat> & frames);
    void ApplyLHEWithInterpol(cv::Mat &base, cv::Mat img, int window);
    void ApplyLHE(cv::Mat &base, cv::Mat img, int window);
};

#endif // SERIALLHE_H_