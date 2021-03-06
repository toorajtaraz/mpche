#ifndef HISTOGRAM_HELPER_H_
#define HISTOGRAM_HELPER_H_

#include <opencv2/opencv.hpp>
int *ExtractHistogram(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt = NULL, int sw = 1);
int *ExtractHistogramRGB(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt = NULL, int sw = 1);
double *CalculateProbability(int *hist, int total_pixels);
double *BuildLookUpTable(double *prob);
double *BuildLookUpTableRGB(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw = true);
#endif // HISTOGRAM_HELPER_H_
