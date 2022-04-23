#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "VideoCreator.h"
#include "ParallelFastLHE.h"
#include "ParallelLHE.h"
#include "SerialLHE.h"
#include <omp.h>

void VideoCreator::videoHandlerPipeline(char *input_path, char *output_path, int color, int mode, int thread_num, int w)
{
    try
    {
        cv::VideoCapture cap(input_path, cv::CAP_FFMPEG);

        if (!cap.isOpened())
        {
            std::cout << "Error: "
                      << "Can not open Video file at " << input_path << std::endl;
        }
        int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(cv::CAP_PROP_FPS), cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

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
                SerialLHE slhe;
                ParallelFastLHE pflhe;
                ParallelLHE plhe;
                cv::Mat base;
                // Window should not be bigger than the image
                if (w > img.cols || w > img.rows)
                {
                    fprintf(stderr, "Error: Window size should not be bigger than the image\n");
                    exit(1);
                }
                // fill base with 0
                base = cv::Mat::zeros(img.size(), CV_MAKETYPE(CV_8U, img.channels()));
                // Swtich case on modes
                switch (mode)
                {
                case 1:
                    // ParallelLHE
                    plhe.ApplyLHE(base, img, w);
                    break;
                case 2:
                    // ParallelFastLHE
                    pflhe.ApplyLHEWithInterpolation(base, img, w);
                    break;
                case 3:
                    // SerialLHE
                    slhe.ApplyLHE(base, img, w);
                    break;
                case 4:
                    // SerialFastLHE
                    slhe.ApplyLHEWithInterpol(base, img, w);
                    break;
                default:
                    break;
                }
#pragma omp critical
                {
                    writer << base;
                }
            }
        }
        writer.release();
    }
    catch (cv::Exception &e)
    {
        std::cout << "error message exception: " << e.msg << std::endl;
        exit(1);
    }
}