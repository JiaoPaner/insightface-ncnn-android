//
// Created by jiaopan on 2019-11-01.
//

#ifndef FACERECOGNITION_UTILS_H
#define FACERECOGNITION_UTILS_H

#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
class Utils {
    public:

        static std::string base64Decode(const char* Data, int DataByte);

        static cv::Mat base64ToMat(std::string &base64_data);

        ~Utils() {

        }
};


#endif //FACERECOGNITION_UTILS_H
