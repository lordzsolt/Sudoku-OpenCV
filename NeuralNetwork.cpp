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
    
    std::vector<float> errors(_trainingLoopCount);
    
    for (int loopCount = 0 ; loopCount < _trainingLoopCount; loopCount++) {
        float error = 0.0f;
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

float NeuralNetwork::learnFromImage(cv::Mat image, int expectedValue) {
    
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
        float outputValue = hyperbolicTangent(s1(neuronIndex, 0), _A, 0);
        y1(neuronIndex, 0) = outputValue;
    }
    
    MatrixXf x2 = y1;
    MatrixXf s2 = _w2.transpose() * x2;
    
    MatrixXf y2(_outputSize, 1);
    for (int neuronIndex = 0 ; neuronIndex < _outputSize ; neuronIndex++) {
        float outputValue = hyperbolicTangent(s2(neuronIndex, 0), _a, 0);
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


//void NeuralNetwork::learnFromImage(cv::Mat image, int expectedValue) {
//    vector<vector<uchar>> input;
//    input.emplace_back(image.data, image.data + _inputSize);
//    for (int i = 0 ; i < _inputSize ; i++) {
//        input[0][i] = input[0][i] > 100;
//    }
//
//    std::vector<float> hiddenLayerValues(_neuronsInHiddenLayer);
//    std::vector<float> hiddenLayerOutput(_neuronsInHiddenLayer);
//    for (int neuronIndex = 0 ; neuronIndex < _neuronsInHiddenLayer ; neuronIndex++) {
//        float value = 0.0f;
//        for (int pixelIndex = 0 ; pixelIndex < _inputSize ; pixelIndex++) {
//            uchar pixelValue = image.at<uchar>(pixelIndex, 1);
//            bool pixelState = pixelValue > 100;
//            value += pixelState * _hiddenLayerWeights[neuronIndex][pixelIndex];
//        }
//        hiddenLayerValues[neuronIndex] = value;
//        hiddenLayerOutput[neuronIndex ] = hyperbolicTangent(value, _activationFunctionSteepness, 0);
//    }
//    
//    std::vector<float> outputLayerValeues(_outputSize);
//    std::vector<float> neuralNetworkOutput(_outputSize);
//    for (int neuronIndex = 0 ; neuronIndex < _neuronsInHiddenLayer ; neuronIndex++) {
//        float value = 0.0f;
//        for (int outputIndex = 0 ; outputIndex < _outputSize ; outputIndex++) {
//            value += hiddenLayerOutput[neuronIndex] * _outputLayerWeights[outputIndex][neuronIndex];
//        }
//        outputLayerValeues[neuronIndex] = value;
//        neuralNetworkOutput[neuronIndex] = hyperbolicTangent(value, _activationFunctionSteepness, 0);
//    }
//
//    std::vector<float> outputDelta(_outputSize);
//    for (int outputIndex = 0 ; outputIndex < _outputSize ; outputIndex++) {
//        outputDelta[outputIndex] = -neuralNetworkOutput[outputIndex];
//        if (outputIndex + 1 == expectedValue) {
//            outputDelta[outputIndex] += 1;
//        }
////        cout << outputIndex << " " << outputDelta[outputIndex];
////        cout << endl;
//    }
//    
//    for (int neuronIndex = 0 ; neuronIndex < _neuronsInHiddenLayer ; neuronIndex++) {
//        
//        vector<float> delta(_outputSize);
//        for (int outputIndex = 0 ; outputIndex < _outputSize ; outputIndex++) {
//            delta[outputIndex] = outputDelta[outputIndex] * hyperbolicTangentDerivative(outputLayerValeues[outputIndex], _activationFunctionSteepness, 0);
//        }
//        
//        
//    }
//    
////    float outputValue = 0.0f;
////    
////    for (int pixelIndex = 0 ; pixelIndex < _inputSize ; pixelIndex++) {
////        outputValue += fValue * _outputLayerWeights[neuronIndex][pixelIndex];
////        cout << outputValue << endl;
////    }
////    
////    float fOutputValue = hyperbolicTangent(outputValue, _activationFunctionSteepness, 0);
//    
//    
//    
//    
//    
////    for (int pixelIndex = 0 ; pixelIndex < _inputSize ;  pixelIndex++) {
////        int row = pixelIndex / 150;
////        int column = pixelIndex % 150;
////        uchar pixelValue = image.at<uchar>(row, column);
////        bool pixelState = pixelValue > 100;
////        
////        float value = 0;
////        for (int neuronIndex = 0; neuronIndex < _neuronsInHiddenLayer ; neuronIndex++) {
////            value += pixelState * _hiddenLayerWeights[pixelIndex][neuronIndex];
//////            float fValue = hyperbolicTangent(value, _activationFunctionSteepness, 0);
////        }
//    
////        float hiddenLayerValue = fValue;
////        float fHiddenLayerValue = hyperbolicTangent(hiddenLayerValue, _activationFunctionSteepness, 0);
////        
////        float deltaHiddenLayer =
////    }
//    
//}


float NeuralNetwork::hyperbolicTangent(float value, float steepness, float theta) {
    float power = -steepness * (value + theta);
    float exponent = exp(power);
    float result = (1 - exponent) / (1 + exponent);
    return result;
}


float NeuralNetwork::hyperbolicTangentDerivative(float value, float steepness, float theta) {
    float power = -steepness * (value + theta);
    float exponent = exp(power);
    float exponentSquared = powf(1 + exponent, 2);
    float result = (2 * steepness * exponent) / exponentSquared;
    return result;
}
