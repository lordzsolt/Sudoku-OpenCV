// Sudoku-OpenCV.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ImageHandler.h"

#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>

static const std::string imageName = "images/4.png";

using namespace cv;

int main() {

	ImageHandler imageHandler(imageName);
	imageHandler.grayscaleFilter();
	imageHandler.integralImage();
//	
//	imageHandler.gaussianBlurFilter(0.5, 3, 3);
//	imageHandler.binaryThresholdFilter(15);
//	imageHandler.cornerHarrisFilter();
	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", imageHandler.lastImage());

	waitKey(0);
    return 0;
}

