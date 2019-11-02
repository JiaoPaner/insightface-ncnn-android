//
// Created by jiaopan on 2019-11-01.
//

#ifndef FACERECOGNITION_RECOGNIZER_H
#define FACERECOGNITION_RECOGNIZER_H

#include "ncnn/net.h"
#include "opencv2/opencv.hpp"

class Recognizer {
    public:
        int loadModel();

        cv::Mat extractFeature(const cv::Mat& face);

        double distance(const cv::Mat& baseMat, const  cv::Mat& targetMat);

        ~Recognizer() {
            net.clear();
        }

    private:
        const int feature_dim = 128;
        ncnn::Net net;
};


#endif //FACERECOGNITION_RECOGNIZER_H
