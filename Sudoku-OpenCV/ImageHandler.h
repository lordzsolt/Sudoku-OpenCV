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

	void grayscaleFilter();
	void rgbToYUV();

	void integralImage();

	void gaussianBlurFilter(float sizePercent, int sigmaColor, int sigmaSpace);
	void binaryThresholdFilter(const int threshold);

	void cornerHarrisFilter();
	void detectCrosses(int crossWidth, int whitespaceWidth);

private:
	std::string _imageName;
	cv::Mat _image;
	cv::Mat _grayscaleImage;

	int filterIndex = 0;
	std::vector<cv::Mat> _filteredImages;

	int smallestSide(const cv::Mat image) const;
	int valueOfAreaInImage();
};

