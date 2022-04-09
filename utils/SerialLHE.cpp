#include "SerialLHE.h"
#include <opencv2/opencv.hpp>
#include <tuple>
#include <iterator>
#include <map>
#include <math.h>
#include <iostream>
const int PIXEL_RANGE = 256;
const int MAX_PIXEL_VAL = 255;

int* SerialLHE::ExtractHistogram(cv::Mat img, int* count, int x_start, int x_end, int y_start, int y_end) {   
    int* hist = new int[PIXEL_RANGE]();
    int height = img.size().height;
    int width = img.size().width;
    if (x_start < 0) x_start = 0;
    if (x_end > height) x_end = height;
    if (y_start < 0) y_start = 0;
    if (y_end > width) y_end = width;

    for (auto i = x_start; i < x_end; i++) {
        for (auto j = y_start; j < y_end; j++) {
            *count += 1;
            hist[img.at<uchar>(i, j)]++;
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
        for (auto j = 0; j <= i; j++) {
            lut[i] += (int) floor(prob[j] * MAX_PIXEL_VAL);
        }
    }
    return lut;
}
void SerialLHE::Test(cv::Mat img) {
    int count = 0;
    int *hist = ExtractHistogram(img, &count, 0, img.size().height, 0, img.size().width);
    double *prob = CalculateProbability(hist, count);
    int *lut = BuildLookUpTable(prob);

    for (auto i = 0; i < PIXEL_RANGE; i++) {
        std::cout << "i = " << i << " val = " << lut[i] << std::endl;
    }
    //create empty base
    cv::Mat base(img.size(), CV_8UC1, cv::Scalar(0));
    this->ApplyLHE(base, img, 151);
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
            std::cout << "(i, j) = (" << i << ", " << j << ")" << std::endl;
            int count = 0;
            int* hist = ExtractHistogram(img, &count, i - offset, i + offset, j - offset, j + offset);
            double* prob = CalculateProbability(hist, count);
            int* lut = BuildLookUpTable(prob);
            all_luts[std::make_tuple(i, j)] = lut;
            delete[] hist;
            delete[] prob;
        }
    }

    //Interpolating local histogram equalization
    int padding_h = (height + ((int) floor((float) window / 2.0) - height % (int) floor((float) window / 2.0))) - height;
    int padding_w = (width + ((int) floor((float) window / 2.0) - width % (int) floor((float) window / 2.0))) - width;

    //Iterate over the image
    for (auto i = 0; i < height; i++) {
        for (auto j = 0; j < width; j++) {
            int x1 = i - (i % (int) floor((float) window / 2.0));
            int y1 = j - (j % (int) floor((float) window / 2.0));
            int x2 = x1 + (int) floor((float) window / 2.0);
            int y2 = y1 + (int) floor((float) window / 2.0);

            float x1_weight = (float) (i - x1) / (float) (x2 - x1);
            float y1_weight = (float) (j - y1) / (float) (y2 - y1);
            float x2_weight = (float) (x2 - i) / (float) (x2 - x1);
            float y2_weight = (float) (y2 - j) / (float) (y2 - y1);

            int* upper_left_lut = all_luts[std::make_tuple(x1, y1)];
            int* upper_right_lut = all_luts[std::make_tuple(x1, y2)];
            int* lower_left_lut = all_luts[std::make_tuple(x2, y1)];
            int* lower_right_lut = all_luts[std::make_tuple(x2, y2)];

            base.at<uchar>(i, j) = (uchar) ceil(upper_left_lut[img.at<uchar>(i, j)] * x2_weight * y2_weight +
                                upper_right_lut[img.at<uchar>(i, j)] * x2_weight * y1_weight +
                                lower_left_lut[img.at<uchar>(i, j)] * x1_weight * y2_weight +
                                lower_right_lut[img.at<uchar>(i, j)] * x1_weight * y1_weight);
        }
    }
    cv::imshow("LHE", base);
    cv::waitKey(0);
    //Cleaning all_luts
    for (auto it = all_luts.begin(); it != all_luts.end(); it++) {
        delete[] it->second;
    }
}
