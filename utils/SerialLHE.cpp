#include "SerialLHE.h"
#include <opencv2/opencv.hpp>
#include <tuple>
#include <iterator>
#include <map>
#include <math.h>
#include <iostream>
const int PIXEL_RANGE = 256;
const int MAX_PIXEL_VAL = 255;

int *SerialLHE::ExtractHistogram(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, int *histt, int sw)
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

int *SerialLHE::ExtractHistogramRGB(cv::Mat img, int *count, int x_start, int x_end, int y_start, int y_end, short channel, int *histt, int sw)
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

double *SerialLHE::CalculateProbability(int *hist, int total_pixels)
{
    double *prob = new double[PIXEL_RANGE]();
    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        prob[i] = (double)hist[i] / total_pixels;
    }
    return prob;
}

double *SerialLHE::BuildLookUpTable(double *prob)
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

double *SerialLHE::BuildLookUpTableRGB(int *hist_blue, int *hist_green, int *hist_red, int count, bool free_sw)
{
    // int count_blue = 0;
    // int *hist_blue = ExtractHistogramRGB(img, &count_blue, 0, img.size().height, 0, img.size().width, 0);
    double *prob_blue = CalculateProbability(hist_blue, count);
    double *lut_blue = BuildLookUpTable(prob_blue);
    delete[] prob_blue;

    // int count_green = 0;
    // int *hist_green = ExtractHistogramRGB(img, &count_green, 0, img.size().height, 0, img.size().width, 1);
    double *prob_green = CalculateProbability(hist_green, count);
    double *lut_green = BuildLookUpTable(prob_green);
    delete[] prob_green;

    // int count_red = 0;
    // int *hist_red = ExtractHistogramRGB(img, &count_red, 0, img.size().height, 0, img.size().width, 2);
    double *prob_red = CalculateProbability(hist_red, count);
    double *lut_red = BuildLookUpTable(prob_red);
    delete[] prob_red;

    double *lut_final = new double[PIXEL_RANGE]();

    for (auto i = 0; i < PIXEL_RANGE; i++)
    {
        lut_final[i] = (lut_blue[i] + lut_green[i] + lut_red[i]) / 3.0;
        // std::cout << lut_final[i] << std::endl;
        // std::cout << "blue: " << lut_blue[i] << std::endl;
        // std::cout << "green: " << lut_green[i] << std::endl;
        // std::cout << "red: " << lut_red[i] << std::endl;
    }
    if (free_sw)
    {
        delete[] lut_blue;
        delete[] lut_green;
        delete[] lut_red;
    }
    return lut_final;
}

void SerialLHE::Test(cv::Mat img,std:: vector< cv::Mat> & frames)
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

    cv::Mat out(img.size().height * 0.15, img.size().width * 0.15, CV_MAKETYPE(CV_8U, img.channels()), cv::Scalar(0));
    cv::resize(img, out, cv::Size(), 0.15, 0.15);
    // print out dimentions
    std::cout << "out.size().height = " << out.size().height << std::endl;
    std::cout << "out.size().width = " << out.size().width << std::endl;
    cv::Mat base(out.size(), CV_MAKETYPE(CV_8U, img.channels()), cv::Scalar(0));
    std::cout << "here" << std::endl;
    // this->ApplyLHEWithInterpol(base, out, 151);
    this->ApplyLHE(base, out, 151);
    // cv::imwrite("/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/base.jpg", base);
    frames.push_back(base);
}

void SerialLHE::ApplyLHE(cv::Mat &base, cv::Mat img, int window)
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
    for (int i = 0; i < height; i++)
    {
        sw = i % 2 == 0 ? 0 : 1;
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
                    double *lut = BuildLookUpTableRGB(hists[0], hists[1], hists[2], count, false);
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
                if (j == 0 && i > 0)
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
                else if (j == 0 && i == 0)
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
                    double *lut = BuildLookUpTableRGB(hists[0], hists[1], hists[2], count, false);
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
        //delete channels
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

void SerialLHE::ApplyLHEWithInterpol(cv::Mat &base, cv::Mat img, int window)
{
    std::map<std::tuple<int, int>, double *> all_luts;
    int offset = (int)floor(window / 2.0);
    int height = img.size().height;
    int width = img.size().width;
    int max_i = height + ((int)floor(window / 2.0) - (height % (int)floor(window / 2.0)));
    int max_j = width + ((int)floor(window / 2.0) - (width % (int)floor(window / 2.0)));
    // get number of channels
    int channels = img.channels();
    std::cout << "channels = " << channels << std::endl;
    for (auto i = 0; i <= max_i; i += offset)
    {
        for (auto j = 0; j <= max_j; j += offset)
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
            all_luts[std::make_tuple(i, j)] = lut;
        }
    }

    // Interpolating local histogram equalization
    int padding_h = (height + ((int)floor((float)window / 2.0) - height % (int)floor((float)window / 2.0))) - height;
    int padding_w = (width + ((int)floor((float)window / 2.0) - width % (int)floor((float)window / 2.0))) - width;

    // Iterate over the image
    for (auto i = 0; i < height; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            int x1 = i - (i % (int)floor((float)window / 2.0));
            int y1 = j - (j % (int)floor((float)window / 2.0));
            int x2 = x1 + (int)floor((float)window / 2.0);
            int y2 = y1 + (int)floor((float)window / 2.0);

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
                    base.at<cv::Vec3b>(i, j)[k] = (uchar)ceil(
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

    // Cleaning all_luts
    for (auto it = all_luts.begin(); it != all_luts.end(); it++)
    {
        delete[] it->second;
    }
}
