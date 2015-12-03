// Sudoku-OpenCV.cpp : Defines the entry point for the console application.
//

#include "ImageHandler.h"

#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>

static const std::string imageName = "../images/binary.png";

static const int kSCREEN_WIDTH = 800;
static const int kSCREEN_HEIGHT = 500;

using namespace cv;

int main() {

	ImageHandler imageHandler(imageName);
//	imageHandler.grayscaleFilter();
//	imageHandler.gaussianBlurFilter(0.5, 3, 3);
//	imageHandler.inverseBinaryThresholdFilter(127);
//	imwrite("../images/binary.png", imageHandler.lastImage());
//	imageHandler.integralImage();
//	imageHandler.detectCorners();
//	imageHandler.cannyFilter();
//	imageHandler.detectCrosses(1,1);
//	
//	imageHandler.gaussianBlurFilter(0.5, 3, 3);
//	imageHandler.cornerHarrisFilter();
	namedWindow("Display window", WINDOW_AUTOSIZE);
	imageHandler.aspectFit(kSCREEN_WIDTH, kSCREEN_HEIGHT);
	auto lastImage = imageHandler.lastImage();

	imshow("Display window", imageHandler.lastImage());

	waitKey(0);
    return 0;
}
