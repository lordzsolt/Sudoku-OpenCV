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

void createTrainingSet();
void trainNeuralNetwork();
void doMagic();


int main() {
//    doMagic();
    trainNeuralNetwork();
    return 0;
}


void createTrainingSet() {
    for (int index = 1 ; index <= 9 ; index++) {
        ostringstream os;
        os << trainingSetFolderPath << "/" << index << ".jpg";
        string imagePath = os.str();
        cout << imagePath;
        
        ImageHandler imageHandler(imagePath);
        imageHandler.preprocessImage();
        imageHandler.findSudokuBoard();
        imageHandler.findSquares();
        
        ostringstream os2;
        os2 << trainingSetFolderPath << "/" << index;
        string savePath = os2.str();
        
        cout << savePath;
        
        imageHandler.saveSquaresTo(savePath);
    }
}


void trainNeuralNetwork() {
    vector<CategorizedImage> images;
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
    
    
    ImageHandler handler(imageName);
    handler.preprocessImage();
    handler.findSudokuBoard();
    handler.findSquares();
    std::vector<cv::Mat> squares = handler.squares();
    
    std::vector<UncategorizedImage> testImages;
    for (int index = 0 ; index < 9 * 9 ; index++) {
        UncategorizedImage image;
        image.image = squares[index];
        testImages.push_back(image);
    }
    
    testImages[2].value = 1;
    testImages[5].value = 7;
    testImages[6].value = 2;
    testImages[10].value = 6;
    testImages[11].value = 4;
    testImages[13].value = 9;
    testImages[14].value = 3;
    testImages[22].value = 8;
    testImages[25].value = 5;
    testImages[29].value = 8;
    testImages[32].value = 4;
    testImages[33].value = 7;
    testImages[35].value = 6;
    testImages[36].value = 4;
    testImages[44].value = 1;
    testImages[45].value = 2;
    testImages[47].value = 3;
    testImages[48].value = 7;
    testImages[51].value = 4;
    testImages[55].value = 3;
    testImages[58].value = 2;
    testImages[66].value = 6;
    testImages[67].value = 3;
    testImages[69].value = 5;
    testImages[70].value = 8;
    testImages[74].value = 2;
    testImages[75].value = 4;
    testImages[78].value = 3;
    
    std::vector<double> a = {1, 0.9, 1.1, 0.75, 1.25};
    std::vector<int> trainingLoops = {300, 500, 750, 1000, 2000};
//    std::vector<int> trainingLoops = {1, 2, 3, 4, 5};
    std::vector<double> A = {0.01, 0.02, 0.025, 0.03, 0.04};
    std::vector<int> neurons = {50, 100, 150, 200, 250, 300};
    std::vector<double> u = {0.01, 0.02, 0.025, 0.03, 0.04};
    
    for (int i = 0 ; i < 5 ; i++) {
        for (int j = 0 ; j < 5 ; j++) {
            for (int k = 0 ; k < 5 ; k++) {
                for (int l = 0 ; l < 5 ; l++) {
                    for (int m = 0 ; m < 5 ; m++) {
                        
                        cout << i << " " << j << " " << k << " " << l << " " << m << endl;
                        
                        NeuralNetwork network(images);
                        
                        network._a = a[i];
                        network._trainingLoopCount = trainingLoops[j];
                        network._A = A[k];
                        network._neuronsInHiddenLayer = neurons[l];
                        network._u = u[m];
                        network.beginLearning();
            
                        network.categorizeImages(testImages);
                    }
                }
            }
        }
    }
}


void doMagic() {
    
    ImageHandler handler(imageName);
    handler.preprocessImage();
    handler.findSudokuBoard();
    handler.findSquares();
    std::vector<cv::Mat> squares = handler.squares();
    
    namedWindow(windowName, WINDOW_AUTOSIZE);
    
    
    imshow(windowName, handler.lastImage());
    waitKey(0);
    
    std::vector<UncategorizedImage> images;
    for (int index = 0 ; index < 9 * 9 ; index++) {
        UncategorizedImage image;
        image.image = squares[index];
        images.push_back(image);
    }
    
    NeuralNetwork network(images);
}