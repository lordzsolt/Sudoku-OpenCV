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
#include <opencv2/highgui.hpp>

struct CategorizedImage {
    int value;
    std::vector<cv::Mat> images;
};

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<CategorizedImage> images);
};

#endif /* NeuralNetwork_hpp */
