#include "ParallelLHE.h"
#include <opencv2/opencv.hpp>
#include <tuple>
#include <iterator>
#include <map>
#include <omp.h>
#include <math.h>
#include <iostream>
const int PIXEL_RANGE = 256;
const int MAX_PIXEL_VAL = 255;

int *ParallelLHE::ExtractHistogram(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt, int sw)
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

int *ParallelLHE::ExtractHistogramRGB(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt, int sw)
{
    if (histt == NULL)
    {
        histt = new int[PIXEL_RANGE]();
    }
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

double *ParallelLHE::CalculateProbability(int *hist, int total_pixels)
{
    double *prob = new double[PIXEL_RANGE]();
    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        prob[i] = (double)hist[i] / total_pixels;
    }
    return prob;
}

double *ParallelLHE::BuildLookUpTable(double *prob)
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

double *ParallelLHE::BuildLookUpTableRGB(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw)
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
    {
        lut_final[i] = (lut_blue[i] + lut_green[i] + lut_red[i]) / 3.0;
    }
    if (free_sw)
    {
        delete[] lut_blue;
        delete[] lut_green;
        delete[] lut_red;
    }
    return lut_final;
}

void ParallelLHE::Test(cv::Mat img)
{
    int count = 0;
    int *hist = ExtractHistogram(img, &count, 0, img.size().height, 0, img.size().width);
    double *prob = CalculateProbability(hist, count);
    double *lut = BuildLookUpTable(prob);

    // for (auto i = 0; i < PIXEL_RANGE; i++)
    // {
    //     std::cout << "i = " << i << " val = " << lut[i] << std::endl;
    // }
    // create empty base
    //  cv::Mat base(img.size(), CV_8UC1, cv::Scalar(0));
    //  this->ApplyLHEWithInterpol(base, img, 251);
    //  cv::imshow("base", base);
    //  cv::waitKey(0);
    float r = 1;
    cv::Mat out(img.size().height * r, img.size().width * r, CV_MAKETYPE(CV_8U, img.channels()), cv::Scalar(0));
    cv::resize(img, out, cv::Size(), r, r);
    // print out dimentions
    std::cout << "out.size().height = " << out.size().height << std::endl;
    std::cout << "out.size().width = " << out.size().width << std::endl;
    cv::Mat base(out.size(), CV_MAKETYPE(CV_8U, img.channels()), cv::Scalar(0));
    std::cout << "here" << std::endl;
    // this->ApplyLHEWithInterpol(base, out, 151);
    this->ApplyLHE(base, out, 251);
    cv::imwrite("/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/base.jpg", base);
}

void ParallelLHE::ApplyLHEHelper(cv::Mat &base, cv::Mat img, int window, int i_start, int i_end)
{
    int offset = (int)floor(window / 2.0);
    int height = img.size().height;
    int width = img.size().width;
    int count = 0;
    int sw = 0;
    int channels = img.channels();
    int **hists;
    int *hist;
    int temp;
    if (channels > 1)
    {
        hists = new int *[channels];
        for (auto i = 0; i < channels; i++)
        {
            hists[i] = new int[PIXEL_RANGE]();
        }
    }
    else
    {
        hist = new int[PIXEL_RANGE]();
    }
    for (int i = i_start; i < i_end; i++)
    {
        sw = i % 2 == (i_start % 2) ? 0 : 1;
        if (sw == 1)
        {
            for (int j = width - 1; j >= 0; j--)
            {
                if (j == (width - 1))
                {
                    for (int n = 0; n < window; n++)
                    {
                        if (channels > 1)
                        {
                            for (auto k = 0; k < channels; k++)
                            {
                                temp = count;
                                ExtractHistogramRGB(img, &temp, i - 1 - offset, i - 1 - offset + 1, j + n - offset, j + n - offset + 1, k, hists[k], -1);
                                ExtractHistogramRGB(img, &temp, i + window - 1 - offset, i + window - 1 - offset + 1, j + n - offset, j + n - offset + 1, k, hists[k], 1);
                            }
                            count = temp;
                        }
                        else
                        {
                            ExtractHistogram(img, &count, i - 1 - offset, i - 1 - offset + 1, j + n - offset, j + n - offset + 1, hist, -1);
                            ExtractHistogram(img, &count, i + window - 1 - offset, i + window - 1 - offset + 1, j + n - offset, j + n - offset + 1, hist, 1);
                        }
                    }
                }
                else if (j < (width - 1))
                {
                    for (int n = 0; n < window; n++)
                    {
                        if (channels > 1)
                        {
                            for (auto k = 0; k < channels; k++)
                            {
                                temp = count;
                                ExtractHistogramRGB(img, &temp, i + n - offset, i + n - offset + 1, j - offset, j - offset + 1, k, hists[k], 1);
                                ExtractHistogramRGB(img, &temp, i + n - offset, i + n - offset + 1, j + window - offset, j + window - offset + 1, k, hists[k], -1);
                            }
                            count = temp;
                        }
                        else
                        {
                            ExtractHistogram(img, &count, i + n - offset, i + n - offset + 1, j - offset, j - offset + 1, hist, 1);
                            ExtractHistogram(img, &count, i + n - offset, i + n - offset + 1, j + window - offset, j + window - offset + 1, hist, -1);
                        }
                    }
                }
                count = count > 0 ? count : 1;
                if (channels > 1)
                {
                    double *lut = BuildLookUpTableRGB(hists[0], hists[1], hists[2], count, true);
                    for (auto k = 0; k < channels; k++)
                    {
                        base.at<cv::Vec3b>(i, j)[k] = (uchar)lut[img.at<cv::Vec3b>(i, j)[k]];
                    }
                    delete[] lut;
                }
                else
                {
                    double *prob = CalculateProbability(hist, count);
                    double *lut = BuildLookUpTable(prob);
                    base.at<uchar>(i, j) = (int)floor(lut[img.at<uchar>(i, j)]);
                    // Clean memory
                    delete[] prob;
                    delete[] lut;
                }
            }
        }
        else
        {
            for (int j = 0; j < width; j++)
            {
                if (j == 0 && i > i_start)
                {
                    for (int n = 0; n < window; n++)
                    {
                        if (channels > 1)
                        {
                            for (auto k = 0; k < channels; k++)
                            {
                                temp = count;
                                ExtractHistogramRGB(img, &temp, i - 1 - offset, i - 1 - offset + 1, j + n - offset, j + n - offset + 1, k, hists[k], -1);
                                ExtractHistogramRGB(img, &temp, i + window - 1 - offset, i + window - 1 - offset + 1, j + n - offset, j + n - offset + 1, k, hists[k], 1);
                            }
                            count = temp;
                        }
                        else
                        {
                            ExtractHistogram(img, &count, i - 1 - offset, i - 1 - offset + 1, j + n - offset, j + n - offset + 1, hist, -1);
                            ExtractHistogram(img, &count, i + window - 1 - offset, i + window - 1 - offset + 1, j + n - offset, j + n - offset + 1, hist, 1);
                        }
                    }
                }
                else if (j == 0 && i == i_start)
                {
                    for (int n = 0; n < window; n++)
                    {
                        for (int m = 0; m < window; m++)
                        {
                            if (channels > 1)
                            {
                                for (auto k = 0; k < channels; k++)
                                {
                                    temp = count;
                                    ExtractHistogramRGB(img, &temp, i + n - offset, i + n - offset + 1, j + m - offset, j + m - offset + 1, k, hists[k], 1);
                                }
                                count = temp;
                            }
                            else
                            {
                                ExtractHistogram(img, &count, i + n - offset, i + n - offset + 1, j + m - offset, j + m - offset + 1, hist, 1);
                            }
                        }
                    }
                }
                else if (j > 0)
                {
                    for (int n = 0; n < window; n++)
                    {
                        if (channels > 1)
                        {
                            for (auto k = 0; k < channels; k++)
                            {
                                temp = count;
                                ExtractHistogramRGB(img, &temp, i + n - offset, i + n - offset + 1, j - 1 - offset, j - 1 - offset + 1, k, hists[k], -1);
                                ExtractHistogramRGB(img, &temp, i + n - offset, i + n - offset + 1, j + window - 1 - offset, j + window - 1 - offset + 1, k, hists[k], 1);
                            }
                            count = temp;
                        }
                        else
                        {
                            ExtractHistogram(img, &count, i + n - offset, i + n - offset + 1, j - 1 - offset, j - 1 - offset + 1, hist, -1);
                            ExtractHistogram(img, &count, i + n - offset, i + n - offset + 1, j + window - 1 - offset, j + window - 1 - offset + 1, hist, 1);
                        }
                    }
                }
                count = count > 0 ? count : 1;
                if (channels > 1)
                {
                    double *lut = BuildLookUpTableRGB(hists[0], hists[1], hists[2], count, true);
                    for (auto k = 0; k < channels; k++)
                    {
                        base.at<cv::Vec3b>(i, j)[k] = (uchar)lut[img.at<cv::Vec3b>(i, j)[k]];
                    }
                    delete[] lut;
                }
                else
                {
                    double *prob = CalculateProbability(hist, count);
                    double *lut = BuildLookUpTable(prob);
                    base.at<uchar>(i, j) = (int)floor(lut[img.at<uchar>(i, j)]);
                    // Clean memory
                    delete[] prob;
                    delete[] lut;
                }
            }
        }
    }
    if (channels > 1)
    {
        // delete channels
        for (auto k = 0; k < channels; k++)
        {
            delete[] hists[k];
        }
    }
    else
    {
        delete[] hist;
    }
}
void ParallelLHE::ApplyLHE(cv::Mat &base, cv::Mat img, int window)
{
// omp_set_num_threads(img.rows / window);
#pragma omp parallel
    {
        int n_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int i_start = thread_id * (base.rows / n_threads);
        int i_end = (thread_id + 1) * (base.rows / n_threads);
#pragma omp critical
        {
            std::cout << "Thread " << thread_id << ": " << i_start << " - " << i_end << std::endl;
        }
        if (thread_id == n_threads - 1)
        {
            i_end = base.rows;
        }
        ApplyLHEHelper(base, img, window, i_start, i_end);
    }
}