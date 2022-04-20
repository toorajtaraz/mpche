#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "headers/SerialLHE.h"
/*TOORAJ INCLUDES BEGIN*/
#include "headers/ParallelLHE.h"
/*TOORAJ INCLUDES END*/

/*ALI INCLUDES BEGIN*/
/*ALI INCLUDES END*/

/*PARIYA INCLUDES BEGIN*/
/*PARIYA INCLUDES END*/
using namespace cv;

/*TOORAJ GLOBALS BEGIN*/
#define T_NUM_THREADS 12
/*TOORAJ GLOBALS END*/

/*ALI GLOBALS BEGIN*/
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
    int t ;
    sscanf(argv[1], "%d", &t);
    omp_set_num_threads(t);
    ParallelLHE plhe;
    SerialLHE slhe;
    Mat img = imread("/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/he8.jpg", 0);
    plhe.Test(img);
    /*TOORAJ END*/

    /*ALI BEGIN*/
    /*ALI END*/

    /*PARIYA BEGIN*/
    /*PARIYA END*/
    return 0;
}
