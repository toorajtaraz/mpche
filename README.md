
# Table of Contents

1.  [Classes](#orgaec21fa)
    1.  [Handling video stream](#orge91cc18)
    2.  [Serial implementation](#orgc226b97)
        1.  [Improved algorithm minus interpolation](#orgaaf2a00)
        2.  [Interpolating implementation](#org1aaa0b7)
    3.  [Parallel implementation](#orga7ffda2)
        1.  [Improved algorithm](#orga94bea1)
        2.  [Interpolating LHE](#org2d595a2)
2.  [Build guide](#orgc15b8cf)
    1.  [Cloning the project](#org00f83ee)
    2.  [Building opencv](#org841a2d0)
    3.  [Building the project](#orgef000b6)
    4.  [Using the command line interface](#orgcd50772)



<a id="orgaec21fa"></a>

# Classes

1.  VideoCreator: Handling video stream input and output.
2.  SerialLHE: Handles serial implementation of local histogram equalization.
3.  ParallelLHE: Handles parallel implementation of improved version local histogram equalization without interpolation.
4.  ParallelFastLHE: Handles parallel implementation of interpolated histogram equalization.
5.  HistogramHelper(NOT A CLASS): Implements utilities for calculating histogram in a window and generate look up table.


<a id="orge91cc18"></a>

## Handling video stream

We simply iterate over frames of input stream and write each manipulated frame to output stream. The iteration for is simply a omp for block and reading the frames and writing them are critical sections. Here we can see the main loop:
```cpp
    #pragma omp parallel num_threads(thread_num)
            {
    #pragma omp for
                for (int frame_num = 0; frame_num < frame_count; frame_num++)
                {
                    cv::Mat img;
    #pragma omp critical
                    {
                        cap >> img;
                    }
                    //Processing the frame
    #pragma omp critical
                    {
                        writer << out;
                    }
                }
            }
```

<a id="orgc226b97"></a>

## Serial implementation


<a id="orgaaf2a00"></a>

### Improved algorithm minus interpolation

The way local histogram equalization works is that we iterate through each pixel in the picture and calculate the histogram for a window centered on that pixel and find the equalized value based on that. Problem with this approach is that every window overlaps with the next window and almost all of it is the same and it doesn&rsquo;t make sense to calculate things over and over again. What we can do is to only calculate the difference between each overlapping window, if we move the window from left to right till the end and then move it down and then move it from right to left we will do the minimum amount of calculation. Alternating the direction we move the window is the key to our problem. Here we can see a simple diagram of the algorithm:

     ─────────►
    
    ┌┬───────┬┐ ┌────────┐               ┌────────┐ │
    ││       ││ │        │   x  x  x     │        │ │
    ││  ┌────┼┼─┼────────┼───────────────┼─────┬──┤ │
    ││  │    ││ │        │               │     │  │ │
    ││  │    ││ │        │               │     │  │ │
    └┴──┼────┴┘ └────────┘               ├─────┼──┤ ▼
        │                                └─────┼──┘
        │                                      │
        │                             ◄────────┼─────
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        │                                      │
        └──────────────────────────────────────┘
```cpp
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
```

<a id="org1aaa0b7"></a>

### Interpolating implementation

Another way to address the problem we talked about earlier is to calculate the histogram around some of the pixels and use interpolation to calculate the mapped value for pixels with no window around them, Using this method we can get pretty good performance even in a single threaded mode. We also can calculate all the look up table before hand and store them in a hashtable, this method is known as dynamic programming.

          ─────────► ◄──────────
    
      ┌───────────┐    ┌───────────┐
      │           │    │           │
      │           │    │           │
      │    ┌──────┼────┼───────────┼──────────────┐
    │ │    │x     │    │     x     │              │
    │ │    │      │    │           │              │
    │ │    │      │    │           │              │
    │ └────┼──────┘    └───────────┘              │
    ▼      │        x                             │
      ┌────┼──────┐    ┌────────────┐             │
    ▲ │    │      │    │            │             │
    │ │    │      │    │            │             │
    │ │    │      │    │            │             │
    │ │    │x     │    │     x      │             │
      │    │      │    │            │             │
      │    │      │    │            │             │
      └────┼──────┘    └────────────┘             │
           │                                      │
           │                                      │
           │                                      │
           │                                      │
           │                                      │
           │                                      │
           │                                      │
           │                                      │
           └──────────────────────────────────────┘
```cpp
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
```

<a id="orga7ffda2"></a>

## Parallel implementation

All we need to do is to divide the work between threads based on their IDs, as no thread writes to same location in memory there is no need for any kind of lock.


<a id="orga94bea1"></a>

### Improved algorithm
```cpp
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
```

<a id="org2d595a2"></a>

### Interpolating LHE
```cpp
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
```

<a id="orgc15b8cf"></a>

# Build guide


<a id="org00f83ee"></a>

## Cloning the project

Opencv is one our project&rsquo;s dependencies, for a better experience and avoiding version related problems we have added a specific commit from opencv project to our project as a submodule, that&rsquo;s why we have to perform a recursive pull.
``` bash
    git clone --recurse-submodules -j[your thread count] https://github.com/toorajtaraz/mpche.git
``` 

<a id="org841a2d0"></a>

## Building opencv

All we need to do is to create a new directory called build, cd into it, have cmake to generate makefile(s) and finally make the whole project. You may want to enable specific features or backends too.
``` bash
    cd mpche/submodules/opencv
    mkdir build
    cd build
    cmake [your flags] ..
    make -j[your thread count]
```


<a id="orgef000b6"></a>

## Building the project

We have to do the same for our project too.
``` bash
    cd mpche
    mkdir build
    cmake ..
    make -j[your thread count]
    ./bin/mpche [flags]
```

<a id="orgcd50772"></a>

## Using the command line interface

All you need to do is to pass &ldquo;-h&rdquo; to our executable to get this help text.

> Usage: ./main -s <is\_stream> -i <input\_path> -o <output\_path> [-t <thread\_num> -r <ratio> -m <mode> -w <window> -c]
> Thread num: Number of threads to be used
> Ratio: Resize ratio for the image (Only in image mode)
> Mode: 1 for PLHE, 2 for FastPLHE, 3 for SLHE and 4 for FastSLHE
> Color: 1 for color, 0 for grayscale
> Stream: 1 for video, 0 for single image
> Window: Size of the window

## Authors
<!-- + Tooraj Taraz ([@ToorajTaraz](https://github.com/ToorajTaraz))
+ Ali Nakhaee ([@alinakhaee](https://github.com/alinakhaee))
+ Pariya AbadehE ([@pariyaab](https://github.com/pariyaab)) -->


| [<img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/64916254?v=4" width="75px;"/><br/>Tooraj Taraz](https://github.com/ToorajTaraz)<br/>  | [<img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/47724896?v=4" width="75px;"/><br/>Ali Nakhaee Sharif](https://github.com/alinakhaee)<br/> | [<img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/46067199?v=4" width="75px;" /><br/>Pariya AbadehE](https://github.com/pariyaab)<br/> |
| :---: | :---: | :---: |