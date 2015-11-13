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
	auto lastImage = this->lastImage();
	Mat result(lastImage.size().width + 1, lastImage.size().height + 1, CV_32S, Scalar(0));

	integral(lastImage, result, CV_32S);
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

void ImageHandler::inverseBinaryThresholdFilter(const int threshold) {
	auto grayscaleImage = this->lastImage();

	Mat result;
	adaptiveThreshold(grayscaleImage, result, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, threshold, 5);
	_filteredImages.push_back(result);
}

void ImageHandler::cannyFilter() {
	auto grayscaleImage = this->lastImage();
	Mat result;
	Canny(grayscaleImage, result, 15, 100, 3);

	_filteredImages.push_back(result);
}

void ImageHandler::detectCorners() {
	auto integralImage = this->lastImage();

	auto patchSize = 3 * 10;

//	for (int patchSize = 3; patchSize <= 64; patchSize *= 3) {
		
		for (auto row = 0; row < integralImage.size().height - patchSize; row++) {
			for (auto column = 0; column < integralImage.size().width - patchSize; column++) {
				auto featureElementSize = patchSize / 3;
				CvPoint topLeft(row, column);
				CvPoint bottomRight(row + featureElementSize, column + featureElementSize);
				auto featueValue = this->valueOfAreaInImage(integralImage, topLeft, bottomRight);
				cerr << featueValue;
				cin.ignore();
			}
		}

//	}
}

void ImageHandler::detectCrosses(int crossWidth, int whitespaceWidth) {

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

