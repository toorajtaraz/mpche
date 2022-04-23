#include "HistogramHelper.h"
#include <opencv2/opencv.hpp>
#include <tuple>
#include <iterator>
#include <map>
#include <omp.h>
#include <math.h>
#include <iostream>
const int PIXEL_RANGE = 256;
const int MAX_PIXEL_VAL = 255;

int *ExtractHistogram(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt, int sw)
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

int *ExtractHistogramRGB(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt, int sw)
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

double *CalculateProbability(int *hist, int total_pixels)
{
    double *prob = new double[PIXEL_RANGE]();
    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        prob[i] = (double)hist[i] / total_pixels;
    }
    return prob;
}

double *BuildLookUpTable(double *prob)
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

double *BuildLookUpTableRGB(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw)
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