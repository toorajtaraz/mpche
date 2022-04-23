#ifndef VIDEO_CREATE_H_
#define VIDEO_CREATE_H_
#include <opencv2/opencv.hpp>

class VideoCreator
{

public:
void videoHandlerPipeline(char* input_path, char* output_path, int color, int mode, int thread_num, int w);
};

#endif // VIDEO_CREATE_H_