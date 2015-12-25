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
    
    std::vector<cv::Mat> squares() {
        return _squares;
    }

	cv::Mat lastImage();
    
    void saveSquaresTo(std::string path);

private:
    std::string _imageName;
    cv::Mat _image;
    
    const int _boardSize = 9;
    const int _outputSize = 50;
    
    cv::Mat _sudokuBoard;
    std::vector<cv::Point> _boardContour;
    cv::Mat _lineSections;
    
    std::vector<cv::Mat> _squares;
    
	std::vector<cv::Mat> _filteredImages;

    void preprocessImage();
    void findSudokuBoard();

    void grayscaleFilter();
    void blurFilter(float sizePercent);
    cv::Mat binaryThresholdFilter(cv::Mat image, const float blockPercent);
    cv::Mat inverseBinaryThresholdFilter(cv::Mat image, const float blockPercent);
    cv::Mat thresholdFunction(cv::Mat image, const float blockPercent, int openCVThresholdType, int inverted);
    
    void findContours();
    void warpBoard(cv::Mat mask);
    void findLines();
    void correctImage();
    
    int smallestSide(const cv::Mat image) const;
    cv::Mat aspectFitImage(cv::Mat image, int screenWidth, int screenHeight);
    void displayImage(cv::Mat image);
};

