#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
using namespace cv;
int main(int argc, char** argv ) {
    #pragma omp parallel
    {
        printf("Hello World... from thread = %d\n",
            omp_get_thread_num());
    }
    std::cout << "Hello, World from non parallel part!\n";
    waitKey(0);
    return 0;
}
