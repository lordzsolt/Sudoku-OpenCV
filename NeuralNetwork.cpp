//
//  NeuralNetwork.cpp
//  Suduku-OpenCV
//
//  Created by Zsolt Kovacs on 12/16/15.
//
//

#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>

using namespace std;
using namespace Eigen;

NeuralNetwork::NeuralNetwork(vector<CategorizedImage> images)
: _trainingSet(images) {
    
    _w1 = MatrixXf::Random(_inputSize, _neuronsInHiddenLayer) * 0.5 * _A;
    _w2 = MatrixXf::Random( _neuronsInHiddenLayer, _outputSize) * 0.5 * _A;
    
    std::vector<double> errors(_trainingLoopCount);
    
    for (int loopCount = 0 ; loopCount < _trainingLoopCount; loopCount++) {
        double error = 0.0f;
        for (int index = 0 ; index < images.size(); index++) {
            for (int imageIndex = 0; imageIndex < _trainingSet[index].images.size(); imageIndex++) {
                auto image = _trainingSet[index].images[imageIndex];
                auto reshapedImage = image.reshape(1, image.cols * image.rows);
                error += learnFromImage(reshapedImage, _trainingSet[index].value);
            }
        }
        cout << loopCount << ": " << error << endl;
    }
}

double NeuralNetwork::learnFromImage(cv::Mat image, int expectedValue) {
    
    MatrixXf d(_outputSize, 1);
    for (int index = 0 ; index < _outputSize ; index++) {
        if (index + 1 == expectedValue) {
            d(index, 0) = 1;
        }
        else {
            d(index, 0) = -1;
        }
    }
    
    MatrixXf x1(_inputSize, 1);
    for (int inputIndex = 0 ; inputIndex < _inputSize ; inputIndex++) {
        unsigned char value = image.at<uchar>(inputIndex, 0);
        x1(inputIndex, 0) =  value > 150;
    }
    
    MatrixXf s1 = _w1.transpose() * x1;
    MatrixXf y1(_neuronsInHiddenLayer, 1);
    for (int neuronIndex = 0 ; neuronIndex < _neuronsInHiddenLayer ; neuronIndex++) {
        double outputValue = hyperbolicTangent(s1(neuronIndex, 0), _a, 0);
        y1(neuronIndex, 0) = outputValue;
    }
    
    MatrixXf x2 = y1;
    MatrixXf s2 = _w2.transpose() * x2;
    
    MatrixXf y2(_outputSize, 1);
    for (int neuronIndex = 0 ; neuronIndex < _outputSize ; neuronIndex++) {
        double outputValue = hyperbolicTangent(s2(neuronIndex, 0), _a, 0);
        y2(neuronIndex, 0) = outputValue;
    }
    
    MatrixXf delta2 = d - y2;
    
    MatrixXf modifier2(_outputSize, 1);
    for (int outputIndex = 0 ; outputIndex < _outputSize ; outputIndex++) {
        modifier2(outputIndex, 0) = delta2(outputIndex, 0) * hyperbolicTangentDerivative(s2(outputIndex, 0), _a, 0);
    }
    
    _w2 += _u * x2 * modifier2.transpose();
    
    MatrixXf delta1 = _w2 * modifier2;
    
    MatrixXf modifier1(_neuronsInHiddenLayer, 1);
    for (int neuronIndex = 0 ; neuronIndex < _neuronsInHiddenLayer ; neuronIndex++) {
        modifier1(neuronIndex, 0) = delta1(neuronIndex, 0) * hyperbolicTangentDerivative(s1(neuronIndex, 0), _a, 0);
    }
    
    _w1 += _u * x1 * modifier1.transpose();
    
    MatrixXf error = delta2.transpose() * delta2;
    return error(0, 0);
}

double NeuralNetwork::hyperbolicTangent(double value, double steepness, double theta) {
    double power = -steepness * (value + theta);
    double exponent = exp(power);
    double result = (1 - exponent) / (1 + exponent);
    return result;
}


double NeuralNetwork::hyperbolicTangentDerivative(double value, double steepness, double theta) {
    double power = -steepness * (value + theta);
    double exponent = exp(power);
    double exponentSquared = powf(1 + exponent, 2);
    double result = (2 * steepness * exponent) / exponentSquared;
    return result;
}
