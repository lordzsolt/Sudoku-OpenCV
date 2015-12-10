// Sudoku-OpenCV.cpp : Defines the entry point for the console application.
//

#include "ImageHandler.h"

#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static const std::string imageName = "../images/6.png";
static const std::string windowName = "Display Window";

static const int kSCREEN_WIDTH = 800;
static const int kSCREEN_HEIGHT = 500;

using namespace cv;

int main() {
    
    namedWindow(windowName, WINDOW_AUTOSIZE);

	ImageHandler imageHandler(imageName);
    imageHandler.preprocessImage();
    imageHandler.findSudokuBoard();
//    imageHandler.correctImage();

	imageHandler.aspectFit(kSCREEN_WIDTH, kSCREEN_HEIGHT);
	imshow(windowName, imageHandler.lastImage());

	waitKey(0);
    destroyAllWindows();
    
    return 0;
}
