#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "headers/SerialLHE.h"
/*TOORAJ INCLUDES BEGIN*/
#include "headers/ParallelLHE.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
/*TOORAJ INCLUDES END*/

/*ALI INCLUDES BEGIN*/
#include "headers/ParallelFastLHE.h"
#include <chrono>
/*ALI INCLUDES END*/

/*PARIYA INCLUDES BEGIN*/
/*PARIYA INCLUDES END*/
using namespace cv;

/*TOORAJ GLOBALS BEGIN*/
#define T_NUM_THREADS 12
/*TOORAJ GLOBALS END*/

/*ALI GLOBALS BEGIN*/
using namespace std::chrono;
/*ALI GLOBALS END*/

/*PARIYA GLOBALS BEGIN*/
/*PARIYA GLOBALS END*/

int main(int argc, char **argv)
{
#pragma omp parallel
    {
        omp_get_num_procs();
    }
    /*PAIR SESS START*/
    int color = 0;
    int bflag = 0;
    char *input_path = NULL;
    char *output_path = NULL;
    int thread_num = 1;
    double ratio = 1.0;
    int mode = 1;
    int is_stream = -1;
    int index;
    int c;

    opterr = 0;

    while ((c = getopt(argc, argv, "hcs:i:o:t:r:m:")) != -1)
        switch (c)
        {
        case 'h':
            //Printing help
            std::cout << "Usage: ./main -s <is_stream> -i <input_path> -o <output_path> [-t <thread_num> -r <ratio> -m <mode> -c]" << std::endl;
            std::cout << "Thread num: Number of threads to be used" << std::endl;
            std::cout << "Ratio: Resize ratio for the image (Only in image mode)" << std::endl;
            std::cout << "Mode: 1 for PLHE, 2 for FastPLHE, 3 for SLHE and 4 for FastSLHE" << std::endl;
            std::cout << "Color: 1 for color, 0 for grayscale" << std::endl;
            std::cout << "Stream: 1 for video, 0 for single image" << std::endl;
            return 0;
            break;
        case 'c':
            color = 1;
            break;
        case 's':
            is_stream = atoi(optarg);
            if (is_stream != 0 && is_stream != 1)
            {
                std::cout << "Error: Invalid stream option, 0 for image and 1 for video" << std::endl;
                exit(1);
            }
            break;
        case 'i':
            input_path = optarg;
            break;
        case 'o':
            output_path = optarg;
            break;
        case 't':
            thread_num = atoi(optarg);
            if (thread_num == 0 || thread_num > omp_get_max_threads() || thread_num < 0) {
                fprintf(stderr, "Error: Thread number must be a positive integer and less than %d\n", omp_get_max_threads());
                exit(1);
            }
            break;
        case 'm':
            mode = atoi(optarg);
            if (mode != 1 && mode != 2 && mode != 3 && mode != 4) {
                fprintf(stderr, "Error: Mode must be 1 for PLHE, 2 for FastPLHE, 3 for SLHE or 4 for FastSLHE\n");
                exit(1);
            }
            break;
        case 'r':
            ratio = strtod(optarg, NULL);
            if (ratio == 0) {
                fprintf(stderr, "ratio must be floating point number\n");
                exit(1); 
            }
            if (ratio > 1) {
                fprintf(stderr, "ratio must be less than 1\n");
                exit(1); 
            }
            break;
        case '?':
            if (optopt == 'i')
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            else if (optopt == 'o')
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            else if (optopt == 't')
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            else if (optopt == 'r')
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr,
                        "Unknown option character `\\x%x'.\n",
                        optopt);
            return 1;
        default:
            abort();
        }

    if (input_path == NULL) {
        fprintf(stderr, "Error: Input path is required\n");
        exit(1);
    }
    if (output_path == NULL) {
        fprintf(stderr, "Error: Output path is required\n");
        exit(1);
    }
    if (is_stream == -1) {
        fprintf(stderr, "Error: Stream option is required\n");
        exit(1);
    }
    for (index = optind; index < argc; index++)
        printf("Invalid argument %s\n", argv[index]);

    std::cout << "Running with these configurations: " << std::endl;
    std::cout << "Input path: " << input_path << std::endl;
    std::cout << "Output path: " << output_path << std::endl;
    std::cout << "Thread number: " << thread_num << std::endl;
    std::cout << "Ratio: " << ratio << std::endl;
    switch (mode) {
        case 1:
            std::cout << "Mode: ParallelLHE" << std::endl;
            break;
        case 2:
            std::cout << "Mode: ParallelFastLHE" << std::endl;
            break;
        case 3:
            std::cout << "Mode: SerialLHE" << std::endl;
            break;
        case 4:
            std::cout << "Mode: SerialFastLHE" << std::endl;
            break;
        default:
            std::cout << "Mode: ParallelLHE" << std::endl;
            break;
    }
    char t_res[6] = {'\0'};
    strncpy(t_res, "False\0", 6);
    if (color != 0) {
        strncpy(t_res, "True\0", 6);
    }
    std::cout << "Has multiple channels: " << t_res << std::endl;
    if (is_stream == 1) {
        std::cout << "Stream: Video" << std::endl;
    } else {
        std::cout << "Stream: Image" << std::endl;
    }
    /*PAIR SESS END*/

    /*TOORAJ BEGIN*/
    // int t ;
    // sscanf(argv[1], "%d", &t);
    // omp_set_num_threads(t);
    // ParallelLHE plhe;
    // SerialLHE slhe;
    // Mat img = imread("/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/he8.jpg", 0);
    // plhe.Test(img);
    /*TOORAJ END*/

    /*ALI BEGIN*/
    // for (auto a_t = 1; a_t <= 16; a_t++)
    // {
    //     auto a_start = high_resolution_clock::now();

    //     omp_set_num_threads(a_t);
    //     ParallelFastLHE a_pflhe;
    //     Mat a_img = imread("/home/mpche/images/tree.jpg", 1);
    //     a_pflhe.Test(a_img);
    //     auto a_stop = high_resolution_clock::now();
    //     auto a_duration = duration_cast<microseconds>(a_stop - a_start);

    //     std::cout << "num of threads: " << a_t << " : " << a_duration.count() / 1000 << " milliseconds" << std::endl;
    // }

    /*ALI END*/

    /*PARIYA BEGIN*/
    /*PARIYA END*/
    return 0;
}
