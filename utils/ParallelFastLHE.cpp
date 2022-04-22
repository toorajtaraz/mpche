#include "ParallelFastLHE.h"
#include <opencv2/opencv.hpp>
#include <tuple>
#include <iterator>
#include <map>
#include <math.h>
#include <iostream>
#include <omp.h>
const int PIXEL_RANGE = 256;
const int MAX_PIXEL_VAL = 255;

int *ParallelFastLHE::ExtractHistogram(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt, int sw)
{
    if (histt == NULL)
    {
        histt = new int[PIXEL_RANGE]();
    }
    else
    {
        if (x_start < 0 || x_end > img.size().height || y_start < 0 || y_end > img.size().width)
        {
            return NULL;
        }
    }
    int height = img.size().height;
    int width = img.size().width;
    if (x_start < 0)
        x_start = 0;
    if (x_end > height)
        x_end = height;
    if (y_start < 0)
        y_start = 0;
    if (y_end > width)
        y_end = width;

    for (auto i = x_start; i < x_end; i++)
    {
        for (auto j = y_start; j < y_end; j++)
        {
            *count += sw;
            histt[img.at<uchar>(i, j)] += sw;
        }
    }
    return histt;
}

int *ParallelFastLHE::ExtractHistogramRGB(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt, int sw)
{
    if (histt == NULL)
        histt = new int[PIXEL_RANGE]();
    else if (x_start < 0 || x_end > img.size().height || y_start < 0 || y_end > img.size().width)
        return NULL;

    int height = img.size().height;
    int width = img.size().width;
    if (x_start < 0)
        x_start = 0;
    if (x_end > height)
        x_end = height;
    if (y_start < 0)
        y_start = 0;
    if (y_end > width)
        y_end = width;

    for (auto i = x_start; i < x_end; i++)
        for (auto j = y_start; j < y_end; j++)
        {
            *count += sw;
            histt[img.at<cv::Vec3b>(i, j)[channel]] += sw;
        }
    return histt;
}

double *ParallelFastLHE::CalculateProbability(int *hist, int total_pixels)
{
    double *prob = new double[PIXEL_RANGE]();
    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        prob[i] = (double)hist[i] / total_pixels;
    }
    return prob;
}

double *ParallelFastLHE::BuildLookUpTable(double *prob)
{
    double *lut = new double[PIXEL_RANGE]();

    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        for (auto j = 0; j <= i; j++)
        {
            lut[i] += prob[j] * MAX_PIXEL_VAL;
        }
    }
    return lut;
}

double *ParallelFastLHE::BuildLookUpTableRGB(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw)
{
    double *prob_blue = CalculateProbability(hist_blue, count);
    double *lut_blue = BuildLookUpTable(prob_blue);
    delete[] prob_blue;

    double *prob_green = CalculateProbability(hist_green, count);
    double *lut_green = BuildLookUpTable(prob_green);
    delete[] prob_green;

    double *prob_red = CalculateProbability(hist_red, count);
    double *lut_red = BuildLookUpTable(prob_red);
    delete[] prob_red;

    double *lut_final = new double[PIXEL_RANGE]();

    for (auto i = 0; i < PIXEL_RANGE; i++)
        lut_final[i] = (lut_blue[i] + lut_green[i] + lut_red[i]) / 3.0;
    if (free_sw)
    {
        delete[] lut_blue;
        delete[] lut_green;
        delete[] lut_red;
    }
    return lut_final;
}

void ParallelFastLHE::Test(cv::Mat img)
{
    cv::Mat base(img.size(), CV_8UC3, cv::Scalar(0));
    this->ApplyLHEWithInterpolation(base, img, 151);
    cv::imwrite("base.jpg", base);
}

void ParallelFastLHE::ApplyLHEWithInterpolHelper(cv::Mat &base, cv::Mat img, int window, int i_start, int i_end, std::map<std::tuple<int, int>, double *> all_luts)
{
    int offset = (int)floor(window / 2.0);
    int height = img.size().height;
    int width = img.size().width;
    int max_i = i_end + (offset - (height % offset));
    int max_j = width + (offset - (width % offset));
    int channels = img.channels();

    // Interpolating local histogram equalization
    int padding_h = (height + (offset - height % offset)) - height;
    int padding_w = (width + (offset - width % offset)) - width;

    // Iterate over the image
    for (auto i = i_start; i < i_end; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            int x1 = i - (i % offset);
            int y1 = j - (j % offset);
            int x2 = x1 + offset;
            int y2 = y1 + offset;

            float x1_weight = (float)(i - x1) / (float)(x2 - x1);
            float y1_weight = (float)(j - y1) / (float)(y2 - y1);
            float x2_weight = (float)(x2 - i) / (float)(x2 - x1);
            float y2_weight = (float)(y2 - j) / (float)(y2 - y1);

            double *upper_left_lut = all_luts[std::make_tuple(x1, y1)];
            double *upper_right_lut = all_luts[std::make_tuple(x1, y2)];
            double *lower_left_lut = all_luts[std::make_tuple(x2, y1)];
            double *lower_right_lut = all_luts[std::make_tuple(x2, y2)];

            if (channels > 1)
            {
                for (auto k = 0; k < channels; k++)
                {

                    base.at<cv::Vec3b>(i, j)[k] = ceil(
                        upper_left_lut[img.at<cv::Vec3b>(i, j)[k]] * x2_weight * y2_weight +
                        upper_right_lut[img.at<cv::Vec3b>(i, j)[k]] * x2_weight * y1_weight +
                        lower_left_lut[img.at<cv::Vec3b>(i, j)[k]] * x1_weight * y2_weight +
                        lower_right_lut[img.at<cv::Vec3b>(i, j)[k]] * x1_weight * y1_weight);
                }
            }
            else
            {
                base.at<uchar>(i, j) = (uchar)ceil(upper_left_lut[img.at<uchar>(i, j)] * x2_weight * y2_weight +
                                                   upper_right_lut[img.at<uchar>(i, j)] * x2_weight * y1_weight +
                                                   lower_left_lut[img.at<uchar>(i, j)] * x1_weight * y2_weight +
                                                   lower_right_lut[img.at<uchar>(i, j)] * x1_weight * y1_weight);
            }
        }
    }
}

void ParallelFastLHE::BuildAllLuts(std::map<std::tuple<int, int>, double *> &all_luts, cv::Mat img, int offset, int i_start, int i_end, int j_start, int j_end)
{
    int channels = img.channels();
    for (auto i = i_start; i < i_end; i++)
    {
        if (i % offset == 0)
        {
            for (auto j = 0; j <= j_end; j += offset)
            {
                int count = 0;
                double *lut;
                if (channels > 1)
                {
                    int **channels_hist = new int *[channels];
                    for (auto k = 0; k < channels; k++)
                    {
                        count = 0;

                        channels_hist[k] = ExtractHistogramRGB(img, &count, i - offset, i + offset, j - offset, j + offset, k);
                    }

                    lut = BuildLookUpTableRGB(channels_hist[2], channels_hist[1], channels_hist[0], count);
                }
                else
                {
                    int *hist = ExtractHistogram(img, &count, i - offset, i + offset, j - offset, j + offset);
                    double *prob = CalculateProbability(hist, count);
                    lut = BuildLookUpTable(prob);
                    delete[] hist;
                    delete[] prob;
                }
#pragma omp critical
                {

                    all_luts[std::make_tuple(i, j)] = lut;
                }
            }
        }
    }
}

void ParallelFastLHE::ApplyLHEWithInterpolation(cv::Mat &base, cv::Mat img, int window)
{
    std::map<std::tuple<int, int>, double *> all_luts;
    int offset = (int)floor(window / 2.0);
    int height = img.size().height;
    int width = img.size().width;
    int max_i = height + (offset - (height % offset));
    int max_j = width + (offset - (width % offset));
    int channels = img.channels();
#pragma omp parallel
    {
        int n_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int i_start = thread_id * (max_i / n_threads);
        int i_end = (thread_id + 1) * (max_i / n_threads);
        if (thread_id == n_threads - 1)
        {
            i_end = max_i + 1;
        }
        BuildAllLuts(all_luts, img, offset, i_start, i_end, 0, max_j);
    }

#pragma omp parallel
    {
        int n_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int i_start = thread_id * (base.rows / n_threads);
        int i_end = (thread_id + 1) * (base.rows / n_threads);
        if (thread_id == n_threads - 1)
        {
            i_end = base.rows;
        }
        ApplyLHEWithInterpolHelper(base, img, window, i_start, i_end, all_luts);
    }

    for (auto it = all_luts.begin(); it != all_luts.end(); it++)
    {
        delete[] it->second;
    }
}
