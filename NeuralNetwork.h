//
//  NeuralNetwork.hpp
//  Suduku-OpenCV
//
//  Created by Zsolt Kovacs on 12/16/15.
//
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <eigen3/Eigen/Dense>

struct CategorizedImage {
    int value;
    std::vector<cv::Mat> images;
};

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<CategorizedImage> images);
    
private:
    std::vector<CategorizedImage> _trainingSet;
    
    const int _trainingLoopCount = 2000;
    
//    const int _inputSize = 150 * 150;
    const int _inputSize = 3 * 3;
//    const int _outputSize = 9;
    const int _outputSize = 4;
    const int _neuronsInHiddenLayer = 10;
    
    const float _a = 1;
    const float _A = 0.01;
    const float _u = 0.01;
    Eigen::MatrixXf _w1;
    Eigen::MatrixXf _w2;
    
    float learnFromImage(cv::Mat image, int expectedValue);
    
    float hyperbolicTangent(float value, float steepness, float theta);
    float hyperbolicTangentDerivative(float value, float steepness, float theta);
    
    void printMatrix(Eigen::MatrixXf matrix);
};

#endif /* NeuralNetwork_hpp */
