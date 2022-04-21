#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "headers/SerialLHE.h"


/*TOORAJ INCLUDES BEGIN*/
/*TOORAJ INCLUDES END*/

/*ALI INCLUDES BEGIN*/
/*ALI INCLUDES END*/

/*PARIYA INCLUDES BEGIN*/
#include "headers/video_extract.h"
/*PARIYA INCLUDES END*/
using namespace cv;

/*TOORAJ GLOBALS BEGIN*/
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
    // SerialLHE slhe;
    // Mat img = imread("/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/he2.jpg");
    // slhe.Test(img);
    /*TOORAJ END*/

    /*ALI BEGIN*/
    /*ALI END*/

    /*PARIYA BEGIN*/

    videoExtract video;
    std::vector<Mat> frames;
    video.extract_frames("C:\\Users\\win10\\Desktop\\video1.mp4",frames);
    std::cout << "finish message: " << "extract frames from video done!" << std::endl;
   
    /*PARIYA END*/
    return 0;
}
