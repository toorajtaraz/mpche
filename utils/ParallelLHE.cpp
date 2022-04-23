#include "ParallelLHE.h"
#include <opencv2/opencv.hpp>
#include "HistogramHelper.h"
#include <tuple>
#include <iterator>
#include <map>
#include <omp.h>
#include <math.h>
#include <iostream>
const int PIXEL_RANGE = 256;
const int MAX_PIXEL_VAL = 255;

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
    cv::Mat base(out.size(), CV_MAKETYPE(CV_8U, img.channels()), cv::Scalar(0));
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
        ApplyLHEHelper(base, img, window, i_start, i_end);
    }
}