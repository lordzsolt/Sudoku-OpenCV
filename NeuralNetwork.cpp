//
//  NeuralNetwork.cpp
//  Suduku-OpenCV
//
//  Created by Zsolt Kovacs on 12/16/15.
//
//

#include "NeuralNetwork.h"

using namespace std;
using namespace Eigen;

NeuralNetwork::NeuralNetwork(vector<CategorizedImage> images)
: _trainingSet(images) {}

NeuralNetwork::NeuralNetwork(vector<UncategorizedImage> images) {
    loadWeights();
    for (int index = 0 ; index < images.size() ; index++) {
        auto image = images[index].image;
        auto reshapedImage = image.reshape(1, image.cols * image.rows);
        images[index].value = categorizeImage(reshapedImage);

        cout << images[index].value << endl;
        
        imshow("Display Window", image);
        cv::waitKey(0);
    }
}


void NeuralNetwork::beginLearning() {
    openLogger();
    loadWeights();
    
    logger << "Loop count: " << _trainingLoopCount << "\n";
    logger << "Hidden Layer: " << _neuronsInHiddenLayer << "\n";
    logger << "_a : " << _a << "\n";
    logger << "_A : " << _A << "\n";
    logger << "_u : " << _u << "\n";
    
    int t0 = time(NULL);
    
    double error = 0.0f;
    for (int loopCount = 0 ; loopCount < _trainingLoopCount; loopCount++) {
        error = 0.0f;
        for (int index = 0 ; index < _trainingSet.size(); index++) {
            for (int imageIndex = 0; imageIndex < _trainingSet[index].images.size(); imageIndex++) {
                auto image = _trainingSet[index].images[imageIndex];
                auto reshapedImage = image.reshape(1, image.cols * image.rows);
                error += learnFromImage(reshapedImage, _trainingSet[index].value);
            }
        }
        if ((loopCount % 25) == 0) {
            logger << loopCount << ": " << error << "\n";
//            cout << loopCount << ": " << error << "\n";
        }
    }
    
    int t1 = time(NULL);
    logger << "Duration: " << t1 - t0 << endl;
    
    saveWeights();
}


void NeuralNetwork::categorizeImages(std::vector<UncategorizedImage> images) {
    loadWeights();
    
    int numberOfCategorizedInputs = 0;
    int numberOfCorrect = 0;
    for (int index = 0 ; index < images.size() ; index++) {
        auto image = images[index].image;
        auto reshapedImage = image.reshape(1, image.cols * image.rows);
        int value = categorizeImage(reshapedImage);
        
        numberOfCategorizedInputs++;
        
        if (value == images[index].value) {
            numberOfCorrect++;
        }
    }
    logger << "Correct: " << numberOfCorrect << "/" << numberOfCategorizedInputs << "\n";
    logger << "Percent: " << (float)numberOfCorrect / numberOfCategorizedInputs << "\n";
    logger.close();
}

void NeuralNetwork::saveWeights() {
    ofstream output;
    output.open(w1FilePath());
    output << _w1.rows() << " " << _w1.cols() << endl;
    for (int row = 0 ; row < _w1.rows() ; row++) {
        for (int col = 0 ; col < _w1.cols() ; col++) {
            output << _w1(row, col) << " ";
        }
        output << "\n";
    }
    output.close();
    
    output.open(w2FilePath());
    output << _w2.rows() << " " << _w2.cols() << endl;
    for (int row = 0 ; row < _w2.rows() ; row++) {
        for (int col = 0 ; col < _w2.cols() ; col++) {
            output << _w2(row, col) << " ";
        }
        output << "\n";
    }
    output.close();
}

void NeuralNetwork::loadWeights() {
    ifstream input;
    input.open(w1FilePath());
    if (input.is_open()) {
        int rows;
        int cols;
        input >> rows >> cols;
        _w1 = MatrixXf(rows, cols);
        for (int row = 0 ; row < rows ; row++) {
            for (int col = 0 ; col < cols ; col++) {
                double value;
                input >> value;
                _w1(row, col) = value;
            }
        }
        input.close();
    }
    else {
        _w1 = MatrixXf::Random(_inputMatrixSize, _neuronsInHiddenLayer) * 0.5 * _A;
    }
    
    input.open(w2FilePath());
    if (input.is_open()) {
        int rows;
        int cols;
        input >> rows >> cols;
        _w2 = MatrixXf(rows, cols);
        for (int row = 0 ; row < rows ; row++) {
            for (int col = 0 ; col < cols ; col++) {
                double value;
                input >> value;
                _w2(row, col) = value;
            }
        }
        input.close();
    }
    else {
        _w2 = MatrixXf::Random( _neuronsInHiddenLayer, _outputSize) * 0.5 * _A;
    }
}


double NeuralNetwork::learnFromImage(cv::Mat image, int expectedValue) {
    MatrixXf d(_outputSize, 1);
    for (int index = 0 ; index < _outputSize ; index++) {
        if (index == expectedValue) {
            d(index, 0) = 1;
        }
        else {
            d(index, 0) = -1;
        }
    }
    
    MatrixXf x1(_inputMatrixSize, 1);
    
    //Bias
    x1(0, 0) = 1;
    
    for (int inputIndex = 0 ; inputIndex < _inputSize ; inputIndex++) {
        unsigned char value = image.at<uchar>(inputIndex, 0);
        x1(inputIndex + 1, 0) =  value > 150;
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


int NeuralNetwork::categorizeImage(cv::Mat image) {
    MatrixXf x1(_inputMatrixSize, 1);
    //Bias
    x1(0, 0) = 1;
    for (int inputIndex = 0 ; inputIndex < _inputSize; inputIndex++) {
        unsigned char value = image.at<uchar>(inputIndex, 0);
        x1(inputIndex + 1, 0) =  value > 150;
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
    
    double maxValue = y2(0, 0);
    int maxIndex = 0;
    
    for (int index = 0 ; index < _outputSize ; index++) {
        if (y2(index, 0) > maxValue) {
            maxValue = y2(index, 0);
            maxIndex = index;
        }
    }
    
    return maxIndex;
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

void NeuralNetwork::openLogger() {
    int index = 0;
    string path;
    do {
        logger.close();
        index++;
        ostringstream os;
        os << outputBasePath << "/" << index << "/";
        finalPath = os.str();
        os << measuresFile;
        path = os.str();
        logger.open(path, ofstream::app);
    }
    while (logger.is_open());
    
    ostringstream os;
    os << "mkdir " << finalPath;
    system(os.str().c_str());
    logger.open(path);
}


std::string NeuralNetwork::w1FilePath() {
//    return "../w1.txt";
    
    ostringstream os;
    os << finalPath << w1OutputFile;
    string path = os.str();
    return path;
}

std::string NeuralNetwork::w2FilePath() {
//    return "../w2.txt";
    
    ostringstream os;
    os << finalPath << w2OutputFile;
    string path = os.str();
    return path;
}
