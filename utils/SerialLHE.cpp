#include "SerialLHE.h"
#include <opencv2/opencv.hpp>
#include <tuple>
#include <iterator>
#include <map>
#include <math.h>
#include <iostream>
const int PIXEL_RANGE = 256;
const int MAX_PIXEL_VAL = 255;

int* SerialLHE::ExtractHistogram(cv::Mat img, int x_start, int x_end, int y_start, int y_end) {   
    int* hist = new int[PIXEL_RANGE]();
    int height = img.size().height;
    int width = img.size().width;
    if (x_start < 0) x_start = 0;
    if (x_end > width) x_end = width;
    if (y_start < 0) y_start = 0;
    if (y_end > height) y_end = height;

    for (auto i = x_start; i < x_end; i++) {
        for (auto j = y_start; j < y_end; j++) {
            // std::cout << pixel << 
            // img.at<uint>(j, i) << "]" << " is " << hist[img.at<uchar>(j, i)];
            hist[img.at<uchar>(j, i)]++;
            // std::cout << "hist[" << img.at<uint>(j, i) << "]" << " is " << hist[img.at<uchar>(j, i)] << std::endl;
        }
    }
    return hist;
}
double* SerialLHE::CalculateProbability(int* hist, int total_pixels) {
    double* prob = new double[PIXEL_RANGE]();
    for (auto i = 0; i < PIXEL_RANGE; i++) {
        prob[i] = (double) hist[i] / total_pixels;
    }
    return prob;
}
int* SerialLHE::BuildLookUpTable(double* prob) {
    int* lut = new int[PIXEL_RANGE]();
    
    for (auto i = 0; i < PIXEL_RANGE; i++) {
        // lut[i] = 0;
        for (auto j = 0; j <= i; j++) {
            lut[i] += (int) floor(prob[j] * MAX_PIXEL_VAL);
        }
    }
    return lut;
}
void SerialLHE::Test(cv::Mat img) {
    int *hist = ExtractHistogram(img, 0, img.size().width, 0, img.size().height);
    double *prob = CalculateProbability(hist, img.size().width * img.size().height);
    int *lut = BuildLookUpTable(prob);

    for (auto i = 0; i < PIXEL_RANGE; i++) {
        std::cout << "i = " << i << " val = " << lut[i] << std::endl;
    }
}
void SerialLHE::ApplyLHE(cv::Mat& base, cv::Mat img, int window) {
    std::map<std::tuple<int, int>, int*> all_luts;
    int offset = (int) floor(window / 2.0);
    int height = img.size().height;
    int width = img.size().width;
    int max_i = height + ((int) floor(window / 2.0) - (height % (int) floor(window / 2.0)));
    int max_j = width + ((int) floor(window / 2.0) - (width % (int) floor(window / 2.0)));
    for (auto i = 0; i <= max_i; i += offset) {
        for (auto j = 0; j <= max_j; j+= offset) {
            int* hist = ExtractHistogram(img, j - offset, j + offset, i - offset, i + offset);
            double* prob = CalculateProbability(hist, (int) (window * window));
            int* lut = BuildLookUpTable(prob);
            all_luts[std::make_tuple(i, j)] = lut;
            delete[] hist;
            delete[] prob;
        }
    }
}
