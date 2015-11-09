#include "stdafx.h"

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

void ImageHandler::grayscaleFilter() {
	Mat grayscaleImage;
	auto lastImage = this->lastImage();
	cvtColor(lastImage, grayscaleImage, CV_BGR2GRAY);
	_filteredImages.push_back(grayscaleImage);
}

void ImageHandler::rgbToYUV() {
	Mat result;
	auto lastImage = this->lastImage();
	cvtColor(lastImage, result, CV_BGR2YCrCb);
	_filteredImages.push_back(result);
}

void ImageHandler::integralImage() {
	Mat result;
	auto lastImage = this->lastImage();

	integral(lastImage, result);
	_filteredImages.push_back(result);
}

void ImageHandler::gaussianBlurFilter(float sizePercent, int sigmaColor, int sigmaSpace) {
	Mat blurredImage;
	auto lastImage = this->lastImage();
	int blockSize = static_cast<int>(this->smallestSide(lastImage) * sizePercent / 100);
	if (blockSize % 2 == 0) {
		blockSize -= 1;
	}
	GaussianBlur(lastImage, blurredImage, CvSize(blockSize, blockSize), sigmaColor, sigmaSpace);
	_filteredImages.push_back(blurredImage);
}

void ImageHandler::binaryThresholdFilter(const int blockSize) {
	auto grayscaleImage = this->lastImage();

	Mat result;
	adaptiveThreshold(grayscaleImage, result, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, blockSize, 5);
	_filteredImages.push_back(result);
}

void ImageHandler::detectCrosses(int crossWidth, int whitespaceWidth) {
	auto lastImage = this->lastImage();

}

void ImageHandler::cornerHarrisFilter() {
	auto lastImage = this->lastImage();

	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	Mat result = Mat::zeros(lastImage.size(), CV_32FC1);
	cornerHarris(lastImage, result, blockSize, apertureSize, k, BORDER_DEFAULT);
	_filteredImages.push_back(result);

	Mat normalisedImage;
	normalize(result, normalisedImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	_filteredImages.push_back(normalisedImage);

	Mat scaledNormalizedImage;
	convertScaleAbs(normalisedImage, scaledNormalizedImage);
	_filteredImages.push_back(scaledNormalizedImage);
}

int ImageHandler::smallestSide(const cv::Mat image) const {
	auto imageHeight = image.size().height;
	auto imageWidth = image.size().width;
	
	return MIN(imageHeight, imageWidth);
}