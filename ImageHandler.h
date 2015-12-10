#pragma once

#include <string>

#include <opencv2/core.hpp>

class ImageHandler
{
public:
	ImageHandler(std::string imageName);
	virtual ~ImageHandler();

	cv::Mat image() {
		return _image;
	}

	cv::Mat lastImage();
    
    void preprocessImage();
    void findSudokuBoard();
    void correctImage();
    
	void aspectFit(int screenWidth, int screenHeight);

private:
	std::string _imageName;
	cv::Mat _image;
    
    cv::Mat _sudokuBoard;
    std::vector<cv::Point> _boardCountour;
    cv::Mat _lineSections;
    
	std::vector<cv::Mat> _filteredImages;

	int smallestSide(const cv::Mat image) const;

    void grayscaleFilter();
    void blurFilter(float sizePercent);
    void binaryThresholdFilter(const float blockPercent);
    void inverseBinaryThresholdFilter(const float blockPercent);
    void thresholdFunction(const float blockPercent, int openCVThresholdType, int inverted);
    
    void findContours();
    void findLines();
    
    
//	template<typename T>
//	T valueOfAreaInImage<T>(const cv::Mat image, const cv::Point topLeft, const cv::Point bottomRight) const {
//		//A B
//		//C D
//		auto a = image.at<T>(topLeft.x, topLeft.y);
//		auto b = image.at<T>(topLeft.y, bottomRight.x);
//		auto c = image.at<T>(topLeft.x, bottomRight.y);
//		auto d = image.at<T>(bottomRight.x, bottomRight.y);
//
//		auto value = d + a - b - c;
//		return value;
//	}

	int32_t valueOfAreaInImage(const cv::Mat image, const cv::Point topLeft, const cv::Point bottomRight) const {
		//A B
		//C D
		auto a = image.at<int32_t>(topLeft.x, topLeft.y);
		auto b = image.at<int32_t>(topLeft.y, bottomRight.x);
		auto c = image.at<int32_t>(topLeft.x, bottomRight.y);
		auto d = image.at<int32_t>(bottomRight.x, bottomRight.y);

		auto value = d + a - b - c;
		return value;
	}
};

