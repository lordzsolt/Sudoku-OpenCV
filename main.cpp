// Sudoku-OpenCV.cpp : Defines the entry point for the console application.
//

#include "ImageHandler.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static const std::string imageName = "../images/2.jpg";
static const std::string trainingSetFolderPath = "../images/learn";

using namespace cv;
using namespace std;

void createTrainingSet();
void trainNeuralNetwork();
void testNetwork();


int main() {
    testNetwork();
//    createTrainingSet();
//    trainNeuralNetwork();
    return 0;
}

void testNetwork() {
    
    ImageHandler handler(imageName);
    std::vector<cv::Mat> squares = handler.squares();
    
    std::vector<UncategorizedImage> images;
    for (int index = 0 ; index < 9 * 9 ; index++) {
        UncategorizedImage image;
        image.image = squares[index];
        images.push_back(image);
    }
    
    NeuralNetwork network(images);
}


void createTrainingSet() {
    for (int index = 0 ; index <= 9 ; index++) {
        ostringstream os;
        os << trainingSetFolderPath << "/" << index << ".jpg";
        string imagePath = os.str();
        cout << imagePath;
        
        ImageHandler imageHandler(imagePath);
        
        ostringstream os2;
        os2 << trainingSetFolderPath << "/" << index;
        string savePath = os2.str();
        
        cout << savePath;
        
        imageHandler.saveSquaresTo(savePath);
    }
}


void trainNeuralNetwork() {
    vector<CategorizedImage> images;
    for (int index = 0; index <= 9; index++) {
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
    
    
    ImageHandler handler("../images/1.jpg");
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
    
//    ImageHandler handler2("../images/3.jpg");
//    std::vector<cv::Mat> squares2 = handler2.squares();
//    
//    for (int index = 0 ; index < 9 * 9 ; index++) {
//        UncategorizedImage image;
//        image.image = squares2[index];
//        testImages.push_back(image);
//    }
//    
//    testImages[81 + 1].value = 2;
//    testImages[81 + 7].value = 6;
//    testImages[81 + 9].value = 4;
//    testImages[81 + 10].value = 8;
//    testImages[81 + 13].value = 7;
//    testImages[81 + 14].value = 6;
//    testImages[81 + 16].value = 1;
//    testImages[81 + 17].value = 3;
//    testImages[81 + 18].value = 3;
//    testImages[81 + 24].value = 8;
//    testImages[81 + 30].value = 6;
//    testImages[81 + 31].value = 4;
//    testImages[81 + 33].value = 9;
//    testImages[80 + 37].value = 6;
//    testImages[81 + 40].value = 2;
//    testImages[81 + 43].value = 3;
//    testImages[81 + 47].value = 9;
//    testImages[81 + 49].value = 3;
//    testImages[81 + 50].value = 1;
//    testImages[81 + 56].value = 4;
//    testImages[81 + 62].value = 1;
//    testImages[81 + 63].value = 8;
//    testImages[81 + 64].value = 1;
//    testImages[81 + 66].value = 7;
//    testImages[81 + 67].value = 9;
//    testImages[81 + 70].value = 4;
//    testImages[81 + 71].value = 2;
//    testImages[81 + 73].value = 7;
//    testImages[81 + 79].value = 8;
    
    
//    std::vector<int> trainingLoops = {300, 300, 300, 300, 300, 300, 300, 300, 300, 500, 750, 1000, 1000};
//    std::vector<int> neurons = {100, 100, 50, 100, 50, 150, 50, 50, 150, 150, 100, 100, 50};
//    std::vector<double> a = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//    std::vector<double> A = {0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.025, 0.025, 0.02, 0.02, 0.015, 0.025};
//    std::vector<double> u = {0.018, 0.025, 0.01, 0.02, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018};

    std::vector<int> trainingLoops = {2500, 10, 10};
    std::vector<int> neurons = {100, 100, 100};
    std::vector<double> a = {1, 1, 1};
    std::vector<double> A = {0.025, 0.015, 0.015};
    std::vector<double> u = {0.018, 0.018, 0.018};

    
    for (int i = 0 ; i < a.size() ; i++) {
        
        cout << i << " " << endl;
        
        NeuralNetwork network(images);
        
        network._a = a[i];
        network._trainingLoopCount = trainingLoops[i];
        network._A = A[i];
        network._neuronsInHiddenLayer = neurons[i];
        network._u = u[i];
        network.beginLearning();
        
        network.categorizeImages(testImages);
    }
}
