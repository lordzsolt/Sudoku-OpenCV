// Sudoku-OpenCV.cpp : Defines the entry point for the console application.
//

#include "ImageHandler.h"

#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static const std::string imageName = "../images/7.jpg";
static const std::string partialResultImage = "../images/binary.png";

static const int kSCREEN_WIDTH = 800;
static const int kSCREEN_HEIGHT = 500;

using namespace cv;

int main() {

	ImageHandler imageHandler(partialResultImage);
	imageHandler.grayscaleFilter();
	imageHandler.gaussianBlurFilter(0.5, 11, 11);
	imageHandler.inverseBinaryThresholdFilter(5);
//	imageHandler.erosionFilter(MORPH_RECT, 3);
//	imageHandler.cannyFilter();
//	imageHandler.findContours();
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

	imshow("Display window", imageHandler.lastImage());

	waitKey(0);
    return 0;
}
