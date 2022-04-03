#ifndef SERIALLHE_H_
#define SERIALLHE_H_
#include <opencv2/opencv.hpp>
class SerialLHE {
    private:
        int* ExtractHistogram(cv::Mat img, int x_start, int x_end, int y_start, int y_end);
        double* CalculateProbability(int* hist, int total_pixels);
        int* BuildLookUpTable(double* prob);
    public:
        void Test(cv::Mat img);
        void ApplyLHE(cv::Mat& base, cv::Mat img, int window);
};

#endif // SERIALLHE_H_