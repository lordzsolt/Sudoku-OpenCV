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
#include <eigen3/Eigen/Core>

struct CategorizedImage {
    int value;
    std::vector<cv::Mat> images;
};

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<CategorizedImage> images);
    
private:
    std::vector<CategorizedImage> _trainingSet;
    
    const int _trainingLoopCount = 500;
    
    const int _inputSize = 50 * 50;
    const int _neuronsInHiddenLayer = 250;
    const int _outputSize = 9;
    
    const double _a = 1;
    const double _A = 0.02;
    const double _u = 0.02;
    Eigen::MatrixXf _w1;
    Eigen::MatrixXf _w2;

    void saveWeights();
    void loadWeights();
    double learnFromImage(cv::Mat image, int expectedValue);
    
    double hyperbolicTangent(double value, double steepness, double theta);
    double hyperbolicTangentDerivative(double value, double steepness, double theta);
    
    template <typename Derived>
    void printMatrix(const Eigen::DenseBase<Derived>& matrix) {
#ifndef DEBUG
        return;
#endif
        for (int row = 0 ; row < matrix.rows() ; row++) {
            for (int col = 0 ; col < matrix.cols() ; col++) {
                std::cout << matrix(row, col) << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif /* NeuralNetwork_hpp */
