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
#include <fstream>
#include <opencv2/highgui.hpp>
#include <Eigen3/Core>
#include <Eigen3/Dense>

struct CategorizedImage {
    int value;
    std::vector<cv::Mat> images;
};

struct UncategorizedImage {
    int value;
    cv::Mat image;
};

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<CategorizedImage> images);
    NeuralNetwork(std::vector<UncategorizedImage> images);
    
    void beginLearning();
    void categorizeImages(std::vector<UncategorizedImage> images);
    
    int _trainingLoopCount = 1000;
    int _neuronsInHiddenLayer = 100;
    double _a = 1;
    double _A = 0.015;
    double _u = 0.018;
    
private:

    std::vector<CategorizedImage> _trainingSet;
    
    static const int _inputSize = 50 * 50;
	static const int _outputSize = 10;
	static const int _inputMatrixSize = _inputSize + 1;
    
    Eigen::MatrixXf _w1;
    Eigen::MatrixXf _w2;
    
    const std::string outputBasePath = "../results";
    const std::string w1OutputFile = "w1.txt";
    const std::string w2OutputFile = "w2.txt";
    const std::string measuresFile = "measure.txt";
    std::string finalPath;
    
    std::ofstream logger;
    void openLogger();
    std::string w1FilePath();
    std::string w2FilePath();

    void saveWeights();
    void loadWeights();
    
    double learnFromImage(cv::Mat image, int expectedValue);
    int categorizeImage(cv::Mat image);
    
    double hyperbolicTangent(double value, double steepness, double theta);
    double hyperbolicTangentDerivative(double value, double steepness, double theta);
    
    template <typename Derived>
    void printMatrix(const Eigen::DenseBase<Derived>& matrix) {
        for (int row = 0 ; row < matrix.rows() ; row++) {
            for (int col = 0 ; col < matrix.cols() ; col++) {
                std::cout << matrix(row, col) << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif /* NeuralNetwork_hpp */
