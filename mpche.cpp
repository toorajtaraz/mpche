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
#include "headers/video_create.h"
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

    // videoExtract video;
    // std::vector<Mat> frames;
    // video.extract_frames("/mpche/images/test.avi",frames);
    // std::cout << "finish message: " << "extract frames from video done!" << std::endl;

    SerialLHE slhe;
    std::vector<Mat> out_put_frames;
    for (int i = 0; i < 3; i++) {
        std::string address_file = "/mpche/images/" + std::to_string(i+1) + ".jpg";
        std::cout <<"address: " << address_file << std::endl;
        Mat img = imread(address_file);
        slhe.Test(img,out_put_frames);
    }
    std::cout << "start to creating video" << std::endl;
    videoCreate videoCreate;
    videoCreate.create_video("/mpche/images/output.mp4",out_put_frames);
    std:: cout << "Finished writing" << std::endl;
    /*PARIYA END*/
    return 0;
}
