#include "ImageHandler.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

static const std::string kWINDOW_NAME = "Display Window";
static const int kSCREEN_WIDTH = 800;
static const int kSCREEN_HEIGHT = 600;

ImageHandler::ImageHandler(std::string imageName)
: _imageName(imageName) {
    
    _image = imread(_imageName);
    if (!_image.data) {
        throw new invalid_argument("Could not find image");
    }
    
    namedWindow(kWINDOW_NAME, WINDOW_AUTOSIZE);
    
    preprocessImage();
    findSudokuBoard();
}

ImageHandler::~ImageHandler() {}

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
    
    //I've tried both with contour and hough lines. Overall I had more success with contours, doe to houghLines finding too many lines
    //thus there were too many intersections between the horizontal and vertical lines.
    //However with contours, sometimes lines would be connected
    findLinesWithContours();
//    findLinesWithHoughLines();
    
    correctImage();
}

void ImageHandler::grayscaleFilter() {
    Mat grayscaleImage;
    auto lastImage = this->lastImage();
    cvtColor(lastImage, grayscaleImage, CV_BGR2GRAY);
    _filteredImages.push_back(grayscaleImage);
    displayImage(grayscaleImage);
}

void ImageHandler::blurFilter(float sizePercent) {
    auto lastImage = this->lastImage();
    
    //Calculate percentage based on smallest side
    int blockSize = static_cast<int>(this->smallestSide(lastImage) * sizePercent / 100);
    if (blockSize % 2 == 0) {
        //Blur area has to be an odd number
        blockSize += 1;
    }
    
    Mat blurredImage;
    blur(lastImage, blurredImage, CvSize(blockSize, blockSize));
    
    _filteredImages.push_back(blurredImage);
    displayImage(blurredImage);
}

cv::Mat ImageHandler::binaryThresholdFilter(cv::Mat image, const float blockPercent) {
    return thresholdFunction(image, blockPercent, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY);
}

cv::Mat ImageHandler::inverseBinaryThresholdFilter(cv::Mat image, const float blockPercent) {
    return thresholdFunction(image, blockPercent, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV);
}

void ImageHandler::findContours() {
    auto originalImage = this->lastImage();
    
    auto lastImage = inverseBinaryThresholdFilter(originalImage, 5);
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(lastImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    
    //We assume the largest area is the Sudoku board
    double largestArea = lastImage.size().height * lastImage.size().width * 0.25;;
    vector<Point> bestContour;
    for (auto it : contours) {
        auto area = contourArea(it);
        if (area > largestArea) {
            largestArea = area;
            bestContour = it;
        }
    }
    
    //Create a mask with the largest area (The board will be white and the rest black)
    Mat mask(lastImage.size().height, lastImage.size().width, lastImage.type(), Scalar(0));
    vector<vector<Point>> contourVector = {bestContour};
    // -1 to fill in the contour
    drawContours(mask, contourVector, 0, Scalar(255), -1);
    
    displayImage(mask);
    
    //Apply mask to cut anything outside the contour
    Mat result;
    bitwise_and(originalImage, mask, result);
    
    _filteredImages.push_back(result);
    _sudokuBoard = result;
    _boardContour = bestContour;
    displayImage(result);
    
    warpBoard(mask);
}


void ImageHandler::warpBoard(cv::Mat mask) {
    auto board = _sudokuBoard;
    Point cornerOne(_sudokuBoard.cols, _sudokuBoard.rows);
    Point cornerTwo(0, 0);
    
    //Starting from both sides find the highest and lowest white points in the mask (these will be 2 corners)
    for (int column = 0 ; column < board.cols ; column++) {
        for (int row = 0 ; row < board.rows ; row++) {
            uchar pixelOne = mask.at<uchar>(row, column);
            
            //Parts belonging to the sudoku board are white (in the mask)
            if (pixelOne > 250) {
                if (cornerOne.x >= column) {
                    cornerOne.x = column;
                    cornerOne.y = row;
                }
            }
            
            int reverseRow = board.rows - row - 1;
            int reverseColumn = board.cols - column - 1;
            uchar pixelTwo = mask.at<uchar>(reverseRow, reverseColumn);
            if (pixelTwo > 250) {
                if (cornerTwo.x <= reverseColumn) {
                    cornerTwo.x = reverseColumn;
                    cornerTwo.y = reverseRow;
                }
            }
        }
        //Stop searching when both corners are found.
        if (cornerOne.x != _sudokuBoard.cols && cornerTwo.x != 0) {
            break;
        }
    }
    
    circle(mask, cornerOne, 25, Scalar(1 * 50), -1);
    circle(mask, cornerTwo, 25, Scalar(2 * 50), -1);
    displayImage(mask);
    
    int sign = cornerOne.y < cornerTwo.y ? -1 : 1;
    
    int deltaY = cornerTwo.y - cornerOne.y;
    int deltaX = cornerTwo.x - cornerOne.x;
    
    //if the delta is less than 3% of the whole board it means finding the corners failed
    //(due to image being taken from a perspective (ex: Looks something like this "/ \"
    //Or it's rotated too much, we're unable to decide the orientation of the board "\" or "/"
    if (abs(deltaY) < abs(deltaX * 0.05)) {
        return;
    }
    
    float arctan = atan2(deltaY, deltaX);
    float angle = arctan * 180 / M_PI + sign * 45.0;
    
    //Rotate by the center of the contour
    auto mom = moments(_boardContour);
    Point centroid;
    centroid.x = mom.m10 /mom.m00;
    centroid.y = mom.m01 / mom.m00;
    
    auto rotationMatrix = getRotationMatrix2D(centroid, angle, 1.0);
    Mat warpedImage;
    warpAffine(_sudokuBoard, warpedImage, rotationMatrix, _sudokuBoard.size());
    warpAffine(_image, _image, rotationMatrix, _image.size());
    
    displayImage(warpedImage);
    _sudokuBoard = warpedImage;
}

void ImageHandler::findLinesWithHoughLines() {
    auto originalImage(_sudokuBoard.clone());
    
    Mat result;
    
    //Binarize the image for houghLines to function correctly.
    Canny(originalImage, result, 10, 45, 3);
    displayImage(result);
    
    vector<Vec2f> houghLines;
    HoughLines(result, houghLines, 1, CV_PI/180, 150);
    
    auto verticalLines = Mat(originalImage.size().height, originalImage.size().width, originalImage.type(), Scalar(0));
    auto horizontalLines = Mat(originalImage.size().height, originalImage.size().width, originalImage.type(), Scalar(0));
    
    for (size_t i = 0; i < houghLines.size(); i++) {
        float rho = houghLines[i][0];
        float theta = houghLines[i][1];
        
        if (theta>CV_PI/180*178 || theta<CV_PI/180*2) {
            //The angle between the left edge of the image and and the lines is 178-180 or 0-2 => it's a vertical line
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + _image.rows*(-b));
            pt1.y = cvRound(y0 + _image.rows*(a));
            pt2.x = cvRound(x0 - _image.rows*(-b));
            pt2.y = cvRound(y0 - _image.rows*(a));
            line(verticalLines, pt1, pt2, Scalar(255), 5);
        }
        else if (theta > CV_PI / 180 * 88 && theta < CV_PI / 180 * 92) {
            //The angle between the left edge of the image and and the lines is 88-92 => it's a horizontal line
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + _image.cols*(-b));
            pt1.y = cvRound(y0 + _image.cols*(a));
            pt2.x = cvRound(x0 - _image.cols*(-b));
            pt2.y = cvRound(y0 - _image.cols*(a));
            line(horizontalLines, pt1, pt2, Scalar(255), 5);
        }
    }
    
    displayImage(horizontalLines);
    displayImage(verticalLines);
    
    Mat section;
    bitwise_and(horizontalLines, verticalLines, section);
    
    displayImage(section);
    _filteredImages.push_back(section);
    _lineSections = section;
}


void ImageHandler::findLinesWithContours() {
    auto originalImage(_sudokuBoard.clone());
    
    auto verticalLines = Mat(originalImage.size().height, originalImage.size().width, originalImage.type(), Scalar(0));
    auto horizontalLines = Mat(originalImage.size().height, originalImage.size().width, originalImage.type(), Scalar(0));
    vector<Mat> lines = {horizontalLines, verticalLines};
    
    //Multipliers will be used for kernel size
    vector<float> widthMultipliers = {0.015, 0.0015};
    vector<float> heightMultipliers = {0.0015, 0.015};
    
    vector<int> sobel = {0, 1};
    
    int contourWidth = boundingRect(_boardContour).size().width;
    int contourHeight = boundingRect(_boardContour).size().height;
    
    for (int i = 0 ; i < 2 ; i++) {
        Mat dx;
        
        //Calculated gradients
        Sobel(originalImage, dx, CV_16S, sobel[i], sobel[1 - i]);
        
        //Convert results back to CV_8U
        convertScaleAbs(dx, dx);
        
        float widthMultiplier = widthMultipliers[i];
        float heightMultiplier = heightMultipliers[i];
        int kernelHeight = originalImage.size().height * heightMultiplier;
        int kernelWidth = originalImage.size().width * widthMultiplier;
        Mat kernel = getStructuringElement(MORPH_RECT, CvSize(kernelWidth, kernelHeight));
        
        //Attept to close lines
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
                //The line must at least half the board to avoid finding very small lines
                isGreaterThanHalf = area.width > contourWidth * 0.5;
                aspectRation = static_cast<float>(area.width) / area.height;
            }
            else {
                //The line must at least half the board to avoid finding very small lines
                isGreaterThanHalf = area.height > contourHeight * 0.5;
                aspectRation = static_cast<float>(area.height) / area.width;
            }
            
            //By default we draw with black (for non-line contours)
            auto color = Scalar(0);
            
            //If the apropriate aspect ratio is greater than 5, then it's a line => We draw the contour with white
            if (aspectRation > 5 && isGreaterThanHalf) {
                color = Scalar(255);
            }
            vector<vector<Point>> contourVector = {it};
            drawContours(mask, contourVector, 0, color, -1);
        }
        
        //Close the contours again in case we found both sides of a single line
        Mat squareKernel = getStructuringElement(MORPH_RECT, CvSize(2, 2));
        morphologyEx(mask, mask, MORPH_CLOSE, squareKernel, Point(-1, -1), 3);
    }
    
    displayImage(horizontalLines);
    displayImage(verticalLines);
    
    //Leave only the intersections between the vertical and horizontal lines
    Mat section;
    bitwise_and(horizontalLines, verticalLines, section);
    
    displayImage(section);
    _filteredImages.push_back(section);
    _lineSections = section;
}

void ImageHandler::correctImage() {
    Mat result(_image.clone());
    auto section = _lineSections;
    
    //Find the contours of the intersections
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(section, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<cv::Point> contourCenters;
    
    auto radius = _image.size().width * 0.01;
    for (auto it : contours) {
        //Calculate the center of mass of the contour
        auto mom = moments(it);
        int x = mom.m10 /mom.m00;
        int y = mom.m01 / mom.m00;
        Point point(x, y);
        contourCenters.push_back(point);
        
        //Draw little dots for debugging
        circle(result, point, radius, Scalar(255, 0, 0), -1);
    }
    
    auto rect = boundingRect(_sudokuBoard);
    
    int boardSize = _boardSize;
    int heightThreshold = rect.size().height / boardSize * 0.25;
    int widthThreshold = rect.size().width / boardSize * 0.25;
    
    vector<Point> filteredPoints;
    for (int i = 0 ; i < contourCenters.size() ; i ++) {
        Point p1 = contourCenters[i];
        bool unique = true;
        for (int j = i + 1 ; j < contourCenters.size() ; j++) {
            Point p2 = contourCenters[j];
            
            //If there are two points in a certain threshold, that means it's a duplicate
            if (abs(p2.x - p1.x) < widthThreshold && abs(p2.y - p1.y) < heightThreshold) {
                unique = false;
                circle(result, p2, radius, Scalar(0, 0, 255), -1);
            }
        }
        if (unique) {
            filteredPoints.push_back(p1);
        }
    }
    
    displayImage(result);
    
    _filteredImages.push_back(result);
    
    //Theoretially we should have exactly 100 points (for a 9x9 board)
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
    
    int squareSize = _outputSize;
    int pointsPerRow = boardSize + 1;
    int edgeInset = (rect.size().width / 9) * 0.11;
    Mat correctedBoard(squareSize * boardSize, squareSize * boardSize, _sudokuBoard.type(), Scalar(0));
    
    //Get each individual square
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
        
        //Attept to warp the image to correct any distortions
        auto transform = getPerspectiveTransform(source, destination);
        Mat warpedImage(squareSize * boardSize, squareSize * boardSize, _sudokuBoard.type(), Scalar(0));
        warpPerspective(_sudokuBoard, warpedImage, transform, Size(squareSize * boardSize, squareSize * boardSize));
        
        Mat smallImage = Mat(warpedImage, Rect(destTopLeft.x, destTopLeft.y, squareSize, squareSize)).clone();
        
        Rect roi(destTopLeft.x, destTopLeft.y, squareSize, squareSize);
        auto binaryImage = binaryThresholdFilter(smallImage, 30);
        binaryImage.copyTo(correctedBoard(roi));
        
        _squares.push_back(binaryImage);
    }
    
    displayImage(correctedBoard);
    
    _filteredImages.push_back(correctedBoard);
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
    blockSize = MAX(blockSize, 3);
    
    Mat result;
    adaptiveThreshold(image, result, 255, openCVThresholdType, inverted, blockSize, 5);
    _filteredImages.push_back(result);
    return result;
}


cv::Mat ImageHandler::aspectFitImage(cv::Mat image, int screenWidth, int screenHeight) {
    //Aspect fit large images before displaying.
    auto widthRatio = 1.0f;
    auto heightRatio = 1.0f;
    if (screenWidth < image.size().width) {
        widthRatio = static_cast<float>(screenWidth) / image.size().width;
    }
    
    if (screenHeight < image.size().height) {
        heightRatio = static_cast<float>(screenHeight) / image.size().width;
    }
    
    auto smallerRatio = MIN(widthRatio, heightRatio);
    
    Size newSize(image.size().width * smallerRatio, image.size().height * smallerRatio);
    Mat result(newSize.height, newSize.width, image.type(), Scalar(0));
    resize(image, result, newSize);
    return result;
}


void ImageHandler::displayImage(cv::Mat image) {
    auto smallImage = aspectFitImage(image, kSCREEN_WIDTH, kSCREEN_HEIGHT);
    imshow(kWINDOW_NAME, smallImage);
    cv::waitKey(0);
}


void ImageHandler::saveSquaresTo(std::string path) {
    for (int index = 0 ; index < _squares.size(); index++) {
        ostringstream os;
        os << path << "/" << index << ".jpg";
        string imagePath = os.str();
        imwrite(imagePath, _squares[index]);
    }
}
