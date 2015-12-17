#include "ImageHandler.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

static const std::string windowName = "Display Window";

static const int kSCREEN_WIDTH = 800;
static const int kSCREEN_HEIGHT = 500;

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

void ImageHandler::correctImage() {
    Mat result(_image.clone());
    auto section = _lineSections;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(section, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<cv::Point> contourCenters;
    
    for (auto it : contours) {
        auto mom = moments(it);
        int x = mom.m10 /mom.m00;
        int y = mom.m01 / mom.m00;
        if (x < 0 || y < 0) {
            continue;
        }
        auto radius = _image.size().width * 0.01;
        Point point(x,y);
        circle(result, point, radius, Scalar(255, 0, 0), -1);
        contourCenters.push_back(point);
    }
    
    auto rect = boundingRect(_sudokuBoard);
    
    int boardSize = 9;
    int heightThreshold = rect.size().height / boardSize * 0.25;
    int widthThreshold = rect.size().width / boardSize * 0.25;
    
    vector<Point> filteredPoints;
    for (int i = 0 ; i < contourCenters.size() ; i ++) {
        Point p1 = contourCenters[i];
        bool unique = true;
        for (int j = i + 1 ; j < contourCenters.size() ; j++) {
            Point p2 = contourCenters[j];
            if (abs(p2.x - p1.x) < widthThreshold && abs(p2.y - p1.y) < heightThreshold) {
                unique = false;
                circle(result, p1, widthThreshold, Scalar(0, 255, 0), -1);
                circle(result, p2, heightThreshold, Scalar(0, 0, 255), -1);
            }
        }
        if (unique) {
            filteredPoints.push_back(p1);
        }
    }
    
    _filteredImages.push_back(result);
    
    if (filteredPoints.size() != 100) {
        cout << "Filtering failed for some reason, there were " << filteredPoints.size() << " points left";
    }
    
    //Sort left->right, top->bottom
    sort(filteredPoints.begin(), filteredPoints.end(), [](const Point& a, const Point& b) -> bool {
        return a.y < b.y;
    });
    for (int i = 0 ; i < 10 ; i++) {
        sort(filteredPoints.begin() + i * 10, filteredPoints.begin() + (i + 1) * 10, [](const Point& a, const Point& b) -> bool {
            return a.x < b.x;
        });
    }
    
    Mat filteredResult(_image.clone());
    for (auto it : filteredPoints) {
        auto radius = _image.size().width * 0.01;
        Point point(it.x,it.y);
        circle(filteredResult, point, radius, Scalar(255, 0, 0), -1);
        contourCenters.push_back(point);
//        
//        _filteredImages.push_back(filteredResult);
//        aspectFit(kSCREEN_WIDTH, kSCREEN_HEIGHT);
//        imshow(windowName, lastImage());
//        
//        waitKey(0);
    }
    
    
    int squareSize = 150;
    int pointsPerRow = boardSize + 1;
    int edgeInset = (rect.size().width / 9) * 0.11;
    Mat correctedBoard(squareSize * boardSize, squareSize * boardSize, _sudokuBoard.type(), Scalar(0));
    for (int i = 0 ; i < filteredPoints.size() ; i++) {
        int row = i / pointsPerRow;
        int column = i % pointsPerRow;
        if (row == 9 || column == 9) {
            continue;
        }
        Point sourceTopLeft = {filteredPoints[i].x + edgeInset, filteredPoints[i].y + edgeInset};
        Point sourceTopRight = {filteredPoints[i + 1].x - edgeInset, filteredPoints[i + 1].y + edgeInset};
        Point sourceBottomLeft = {filteredPoints[i + pointsPerRow].x + edgeInset, filteredPoints[i + pointsPerRow].y - edgeInset};
        Point sourceBottomRight = {filteredPoints[i + 1 + pointsPerRow].x - edgeInset, filteredPoints[i + 1 + pointsPerRow].y - edgeInset};
        
        Point destTopLeft(column * squareSize, row * squareSize);
        Point destTopRight((column + 1) * squareSize, row * squareSize);
        Point destBottomLeft(column * squareSize, (row + 1) * squareSize);
        Point destBottomRight((column + 1) * squareSize, (row + 1) * squareSize);
        
        Point2f source[] = {sourceTopLeft, sourceTopRight, sourceBottomLeft, sourceBottomRight};
        Point2f destination[] = {destTopLeft, destTopRight, destBottomLeft, destBottomRight};
        
        auto transform = getPerspectiveTransform(source, destination);
        Mat warpedImage(squareSize * boardSize, squareSize * boardSize, _sudokuBoard.type(), Scalar(0));
        warpPerspective(_sudokuBoard, warpedImage, transform, Size(squareSize * boardSize, squareSize * boardSize));
     
        Mat smallImage = Mat(warpedImage, Rect(destTopLeft.x, destTopLeft.y, squareSize, squareSize)).clone();
        
        Rect roi(destTopLeft.x, destTopLeft.y, squareSize, squareSize);
        auto binaryImage = binaryThresholdFilter(smallImage, 30);
        binaryImage.copyTo(correctedBoard(roi));
        
        _squares.push_back(binaryImage);
    }
    _filteredImages.push_back(correctedBoard);
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

cv::Mat ImageHandler::binaryThresholdFilter(cv::Mat image, const float blockPercent) {
    return thresholdFunction(image, blockPercent, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY);
}

cv::Mat ImageHandler::inverseBinaryThresholdFilter(cv::Mat image, const float blockPercent) {
    return thresholdFunction(image, blockPercent, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV);
}

void ImageHandler::findContours() {
    auto originalImage = this->lastImage();
    
    //TODO: Find out why it doesn't work if I remove this function call
    auto lastImage = inverseBinaryThresholdFilter(originalImage, 5);
    
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
    
    int contourWidth = boundingRect(_boardCountour).size().width;
    int contourHeight = boundingRect(_boardCountour).size().height;
    
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
            bool isGreaterThanHalf = false;
            if (i == 0) {
                isGreaterThanHalf = area.width > contourWidth * 0.5;
                aspectRation = static_cast<float>(area.width) / area.height;
            }
            else {
                isGreaterThanHalf = area.height > contourHeight * 0.5;
                aspectRation = static_cast<float>(area.height) / area.width;
            }
            vector<vector<Point>> contourVector = {it};
            auto color = Scalar(0);
            if (aspectRation > 4 && isGreaterThanHalf) {
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

    Size newSize(lastImage.size().width * smallerRatio, lastImage.size().height * smallerRatio);
    Mat result(newSize.height, newSize.width, lastImage.type(), Scalar(0));
    resize(lastImage, result, newSize);
    _filteredImages.push_back(result);
}

int ImageHandler::smallestSide(const cv::Mat image) const {
    auto imageHeight = image.size().height;
    auto imageWidth = image.size().width;
    
    return MIN(imageHeight, imageWidth);
}

cv::Mat ImageHandler::thresholdFunction(cv::Mat image, const float blockPercent, int openCVThresholdType, int inverted) {
    int blockSize = floor(this->smallestSide(image) * blockPercent / 100);
    if (blockSize % 2 == 0) {
        blockSize--;
    }
    if (blockSize < 3) {
        blockSize = 3;
    }
    
    Mat result;
    adaptiveThreshold(image, result, 255, openCVThresholdType, inverted, blockSize, 5);
    _filteredImages.push_back(result);
    return result;
}
