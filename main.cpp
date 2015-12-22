// Sudoku-OpenCV.cpp : Defines the entry point for the console application.
//

#include "ImageHandler.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static const std::string imageName = "../images/6.png";
static const std::string trainingSetFolderPath = "../images/learn";
static const std::string windowName = "Display Window";

static const int kSCREEN_WIDTH = 800;
static const int kSCREEN_HEIGHT = 500;

using namespace cv;
using namespace std;

int main() {
    
    vector<CategorizedImage> images(4);
    for (int index = 1; index <= 9; index++) {
        CategorizedImage image;
        image.value = index;
        for (int i = 0; i < 81; i++) {
            ostringstream os;
            os << trainingSetFolderPath << "/" << index << "/" << i << ".jpg";
            string imagePath = os.str();
            auto mat = imread(imagePath, IMREAD_GRAYSCALE);
            image.images.push_back(mat);
        }
        images.push_back(image);
    }
    
//    images[0].value = 1;
//    images[1].value = 2;
//    images[2].value = 3;
//    images[3].value = 4;
//    
//    Mat image1(3, 3, CV_8UC1);
//    image1.at<uchar>(0, 0) = 255;
//    image1.at<uchar>(0, 1) = 255;
//    image1.at<uchar>(0, 2) = 255;
//    image1.at<uchar>(0, 3) = 0;
//    image1.at<uchar>(0, 4) = 255;
//    image1.at<uchar>(0, 5) = 0;
//    image1.at<uchar>(0, 6) = 0;
//    image1.at<uchar>(0, 7) = 255;
//    image1.at<uchar>(0, 7) = 0;
//    images[0].images.push_back(image1);
//    
//    Mat image2(3, 3, CV_8UC1);
//    image2.at<uchar>(0, 0) = 255;
//    image2.at<uchar>(0, 1) = 0;
//    image2.at<uchar>(0, 2) = 255;
//    image2.at<uchar>(0, 3) = 255;
//    image2.at<uchar>(0, 4) = 255;
//    image2.at<uchar>(0, 5) = 255;
//    image2.at<uchar>(0, 6) = 255;
//    image2.at<uchar>(0, 7) = 0;
//    image2.at<uchar>(0, 7) = 255;
//    images[1].images.push_back(image2);
//    
//    Mat image3(3, 3, CV_8UC1);
//    image3.at<uchar>(0, 0) = 255;
//    image3.at<uchar>(0, 1) = 255;
//    image3.at<uchar>(0, 2) = 255;
//    image3.at<uchar>(0, 3) = 255;
//    image3.at<uchar>(0, 4) = 0;
//    image3.at<uchar>(0, 5) = 0;
//    image3.at<uchar>(0, 6) = 255;
//    image3.at<uchar>(0, 7) = 0;
//    image3.at<uchar>(0, 7) = 0;
//    images[2].images.push_back(image3);
//    
//    Mat image4(3, 3, CV_8UC1);
//    image4.at<uchar>(0, 0) = 0;
//    image4.at<uchar>(0, 1) = 0;
//    image4.at<uchar>(0, 2) = 255;
//    image4.at<uchar>(0, 3) = 0;
//    image4.at<uchar>(0, 4) = 0;
//    image4.at<uchar>(0, 5) = 255;
//    image4.at<uchar>(0, 6) = 255;
//    image4.at<uchar>(0, 7) = 255;
//    image4.at<uchar>(0, 7) = 255;
//    images[3].images.push_back(image4);
    
    NeuralNetwork network(images);
    
    
//    namedWindow(windowName, WINDOW_AUTOSIZE);
    
    
//	ImageHandler imageHandler(imageName);
//    imageHandler.preprocessImage();
//    imageHandler.findSudokuBoard();
//    imageHandler.findSquares();

//	waitKey(0);
//    desstroyAllWindows();
    
    return 0;
}
