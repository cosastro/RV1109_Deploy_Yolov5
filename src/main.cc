#include "objectDet.hpp"
#include <stdio.h>
using namespace std;

int main(int argc, char **argv){
  // Initialize RKNN model
  const char * model_path = argv[1];
  ObjDet fire;
  fire.Init(model_path);  
  
  //load input image
  cv::Mat input_image  = cv::imread(argv[2]);
  
  //define detection results
  std::vector<cv::Rect> rects;

  //do processing
  fire.DetProcess(input_image, rects);
}