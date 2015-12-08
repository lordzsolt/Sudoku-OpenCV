#include "ImageHandler.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

ImageHandler::ImageHandler(std::string imageName)
: _imageName(imageName) {
    
    _image = imread(_imageName);
    if (!_image.data) {
        std::cout << "Could not open or find the image" << std::endl;
    }
}

ImageHandler::~ImageHandler() {
}

cv::Mat ImageHandler::lastImage() {
    if (_filteredImages.size() == 0) {
        return _image;
    }
    return _filteredImages.back();
}

void ImageHandler::preprocessImage() {
    grayscaleFilter();
    blurFilter(0.5);
}


void ImageHandler::findSudokuBoard() {
    findContours();
    findLines();
}

void ImageHandler::grayscaleFilter() {
    Mat grayscaleImage;
    auto lastImage = this->lastImage();
    cvtColor(lastImage, grayscaleImage, CV_BGR2GRAY);
    _filteredImages.push_back(grayscaleImage);
}

void ImageHandler::blurFilter(float sizePercent) {
    auto lastImage = this->lastImage();
    Mat blurredImage(lastImage.size().height, lastImage.size().width, lastImage.type(), Scalar(0));
    int blockSize = static_cast<int>(this->smallestSide(lastImage) * sizePercent / 100);
    if (blockSize % 2 == 0) {
        blockSize -= 1;
    }
    blur(lastImage, blurredImage, CvSize(blockSize, blockSize));
    _filteredImages.push_back(blurredImage);
}

void ImageHandler::binaryThresholdFilter(const float blockPercent) {
    thresholdFunction(blockPercent, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY);
}

void ImageHandler::inverseBinaryThresholdFilter(const float blockPercent) {
    thresholdFunction(blockPercent, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV);
}

void ImageHandler::findContours() {
    auto originalImage = this->lastImage();
    
    //TODO: Find out why it doesn't work if I remove this function call
    inverseBinaryThresholdFilter(5);
    auto lastImage = this->lastImage();
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    cv::findContours(lastImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    
    double largestArea = lastImage.size().height * lastImage.size().width * 0.25;;
    vector<Point> bestContour;
    for (auto it : contours) {
        auto area = contourArea(it);
        if (area > largestArea) {
            largestArea = area;
            bestContour = it;
        }
    }
    
    Mat mask(lastImage.size().height, lastImage.size().width, lastImage.type(), Scalar(0));
    vector<vector<Point>> contourVector = {bestContour};
    drawContours(mask, contourVector, 0, Scalar(255), -1);
    int contourInset = MIN(boundingRect(bestContour).size().width, boundingRect(bestContour).size().height) * 0.05;
    drawContours(mask, contourVector, 0, Scalar(0), contourInset);
    
    Mat result;
    bitwise_and(originalImage, mask, result);
    _filteredImages.push_back(result);
    _sudokuBoard = result;
    _boardCountour = bestContour;
    
}

void ImageHandler::findLines() {
    auto originalImage(_sudokuBoard.clone());
    
    auto lineOne = Mat(originalImage.size().height, originalImage.size().width, originalImage.type(), Scalar(0));
    auto lineTwo = Mat(originalImage.size().height, originalImage.size().width, originalImage.type(), Scalar(0));
    
    vector<Mat> lines = {lineOne, lineTwo};
    vector<float> widthMultipliers = {0.02, 0.002};
    vector<float> heightMultipliers = {0.002, 0.02};
    vector<int> sobel = {0, 1};
    
    for (int i = 0 ; i < 2 ; i++) {
        float widthMultiplier = widthMultipliers[i];
        float heightMultiplier = heightMultipliers[i];
        int kernelHeight = originalImage.size().height * heightMultiplier;
        int kernelWidth = originalImage.size().width * widthMultiplier;
    
        Mat kernel = getStructuringElement(MORPH_RECT, CvSize(kernelWidth, kernelHeight));
        Mat dx;
        
        Sobel(originalImage, dx, CV_16S, sobel[i], sobel[1 - i]);
        convertScaleAbs(dx, dx);
        
        Mat close;
        threshold(dx, close, 0, 255, THRESH_BINARY | THRESH_OTSU);
        morphologyEx(close, close, MORPH_DILATE, kernel);
    
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        cv::findContours(close, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        auto mask = lines[i];
        for (auto it : contours) {
            auto area = boundingRect(it);
            float aspectRation = 0.0f;
            if (i == 0) {
                aspectRation = static_cast<float>(area.width) / area.height;
            }
            else {
                aspectRation = static_cast<float>(area.height) / area.width;
            }
            vector<vector<Point>> contourVector = {it};
            auto color = Scalar(0);
            if (aspectRation > 4) {
                color = Scalar(255);
            }
            drawContours(mask, contourVector, 0, color, -1);
        }
        
        Mat squareKernel = getStructuringElement(MORPH_RECT, CvSize(2, 2));
        morphologyEx(mask, mask, MORPH_CLOSE, squareKernel, Point(-1, -1), 3);
    }
    
    Mat horizontalLines = lines[0];
    Mat verticalLines = lines[1];
    
    Mat section;
    bitwise_and(horizontalLines, verticalLines, section);
    
    _filteredImages.push_back(section);
    _lineSections = section;
}

void ImageHandler::aspectFit(int screenWidth, int screenHeight) {
    auto lastImage = this->lastImage();
    auto widthRatio = 1.0f; 
    auto heightRatio = 1.0f;
    if (screenWidth < lastImage.size().width) {
        widthRatio = static_cast<float>(screenWidth) / lastImage.size().width;
    }
    
    if (screenHeight < lastImage.size().height) {
        heightRatio = static_cast<float>(screenHeight) / lastImage.size().width;
    }
    
    auto smallerRatio = MIN(widthRatio, heightRatio);
    
    Mat result;
    Size newSize(lastImage.size().width * smallerRatio, lastImage.size().height * smallerRatio);
    resize(lastImage, result, newSize);
    _filteredImages.push_back(result);
}

int ImageHandler::smallestSide(const cv::Mat image) const {
    auto imageHeight = image.size().height;
    auto imageWidth = image.size().width;
    
    return MIN(imageHeight, imageWidth);
}

void ImageHandler::thresholdFunction(const float blockPercent, int openCVThresholdType, int inverted) {
    auto grayscaleImage = this->lastImage();
    int blockSize = floor(this->smallestSide(grayscaleImage) * blockPercent / 100);
    if (blockSize % 2 == 0) {
        blockSize--;
    }
    
    Mat result;
    adaptiveThreshold(grayscaleImage, result, 255, openCVThresholdType, inverted, blockSize, 5);
    _filteredImages.push_back(result);
}
