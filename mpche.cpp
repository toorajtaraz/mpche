#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "headers/SerialLHE.h"
/*TOORAJ INCLUDES BEGIN*/
/*TOORAJ INCLUDES END*/

/*ALI INCLUDES BEGIN*/
#include "headers/ParallelFastLHE.h"
#include <chrono>
/*ALI INCLUDES END*/

/*PARIYA INCLUDES BEGIN*/
/*PARIYA INCLUDES END*/
using namespace cv;

/*TOORAJ GLOBALS BEGIN*/
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
    /*PAIR SESS END*/

    /*TOORAJ BEGIN*/
    /*TOORAJ END*/

    /*ALI BEGIN*/
    for (auto t = 1; t <= 16; t++)
    {
        auto start = high_resolution_clock::now();

        omp_set_num_threads(t);
        ParallelFastLHE pflhe;
        Mat img = imread("/home/mpche/images/tree.jpg", 1);
        pflhe.Test(img);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        std::cout << "num of threads: " << t << " : " << duration.count() / 1000 << " milliseconds" << std::endl;
    }

    /*ALI END*/

    /*PARIYA BEGIN*/
    /*PARIYA END*/
    return 0;
}
