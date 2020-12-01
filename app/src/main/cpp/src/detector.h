//
// Created by jiaopan on 2019-11-02.
//

#ifndef FACERECOGNITION_DETECTOR_H
#define FACERECOGNITION_DETECTOR_H
#include <stdio.h>
#include <vector>
#include "ncnn/net.h"
#include "opencv2/opencv.hpp"
#include "ncnn/platform.h"

struct FaceObject{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
    int type;
};
class Detector {
    public:
        int loadModel();
        int loadMaskModel();

        float intersection_area(const FaceObject& a, const FaceObject& b);

        void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right);

        void qsort_descent_inplace(std::vector<FaceObject>& faceobjects);

        void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold);

        ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales);

        void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, float prob_threshold, std::vector<FaceObject>& faceobjects);
        void generate_mask_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob,const ncnn::Mat& type_blob,
                                     float prob_threshold,float mask_threshold,std::vector<FaceObject>& faceobjects);

        void detect_retinaface(const cv::Mat& bgr, std::vector<FaceObject>& faceobjects);
        void detect_maskface(const cv::Mat& bgr, std::vector<FaceObject>& faceobjects);

        std::vector<cv::Mat> createAlignFace(cv::Mat& img,int type);
    ~Detector() {
        net.clear();
    }
    private:
        ncnn::Net net;
};


#endif //FACERECOGNITION_DETECTOR_H
