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

void ImageHandler::erosionFilter(int erosionType, int erosionSize) {
	
	auto lastImage = this->lastImage();

	Mat element = getStructuringElement(erosionType,
		Size(2 * erosionSize + 1, 2 * erosionSize + 1),
		Point(erosionSize, erosionSize));

	Mat result;
	/// Apply the erosion operation
	erode(lastImage, result, element);

	_filteredImages.push_back(result);
}

void ImageHandler::binaryThresholdFilter(const float blockPercent) {
	thresholdFunction(blockPercent, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY);
}

void ImageHandler::inverseBinaryThresholdFilter(const float blockPercent) {
	thresholdFunction(blockPercent, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV);
}

void ImageHandler::cannyFilter() {
	auto grayscaleImage = this->lastImage();
	Mat result;
	Canny(grayscaleImage, result, 15, 2 * 15, 3);

	_filteredImages.push_back(result); 
}

void ImageHandler::detectCorners() {
	auto integralImage = this->lastImage();

	auto patchSize = 3 * 10;

//	for (int patchSize = 3; patchSize <= 64; patchSize *= 3) {
		
	Mat result(integralImage.size().width, integralImage.size().height, CV_32S, Scalar(0));

	auto index = 0;

		for (auto row = 0; row < integralImage.size().width - patchSize - 20; row++) {
			for (auto column = 0; column < integralImage.size().height - patchSize - 20; column++) {
				auto featureElementSize = patchSize / 3;
				CvPoint topLeft(row, column);
				int multipliers[3][3] = { {0, 0, 0}, {0, 1, 1}, {0, 1, 0} };
				auto featueValue = this->computeHaarFeature(integralImage, featureElementSize, multipliers, topLeft);
				if (featueValue > 300000000) {
					result.at<int32_t>(row, column) = 255;
//					cout << ++index << ": " << featueValue << "\r";
//					cin.ignore();
				}
//				cin.ignore();
			}
		}


			imwrite("../images/haarCorner.png", result);
//	}
}

void ImageHandler::detectCrosses(int crossWidth, int whitespaceWidth) {

}

void ImageHandler::findContours() {
	auto lastImage = this->lastImage();

	cv::findContours(lastImage, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat result = Mat::zeros(lastImage.size(), CV_8UC3);

	for (int i = 0; i< contours.size(); i++) {
		auto color = Scalar(0, 255, 255);
		drawContours(result, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	_filteredImages.push_back(result);
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

int ImageHandler::computeHaarFeature(const cv::Mat& image, const int patchSize, const int multipliers[3][3], cv::Point origin) {
	auto sum = 0;

	for (auto row = 0; row < 3; row++) {
		for (auto column = 0; column < 3; column++) {
			Point topLeft(origin.x + row * patchSize, origin.y + column * patchSize);
			Point bottomRight(origin.x + (row + 1) * patchSize, origin.y + (column + 1) * patchSize);
			sum += 255 * multipliers[row][column] * patchSize - valueOfAreaInImage(image, topLeft, bottomRight);
		}
	}

	return sum;
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