//
// Created by jiaopan on 2019-11-01.
//

#include "recognizer.h"
#include "detface.mem.h"
#include "detface.id.h"

int Recognizer::loadModel() {
    int param = this->net.load_param(detface_param_bin);
    int model = this->net.load_model(detface_bin);
    if(param > 0 && model > 0){
        return 1;
    }
    return 0;
}

cv::Mat Recognizer::extractFeature(const cv::Mat &face) {
    std::vector<float> feature;
    cv::Mat output;
    feature.resize(this->feature_dim);
    if(face.empty()){
        return output;
    }
    cv::Mat image = face.clone();
    cv::resize(image, image, cv::Size(112, 112));

    ncnn::Extractor ex = this->net.create_extractor();
    ex.set_light_mode(true);
    //ex.set_num_threads(4);
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(face.data, ncnn::Mat::PIXEL_BGR, face.cols, face.rows, 112, 112);
    ncnn::Mat in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);

    ncnn::Mat out;
    ex.input(detface_param_id::BLOB_data, in);
    int status = ex.extract(detface_param_id::BLOB_fc1, out);

    if(status == 0){
        for (int i = 0; i < this->feature_dim; i++){
            feature[i] = out[i];
        }
        output = cv::Mat(feature, true).reshape(1, 1);
        cv::normalize(output, output);
    }
    return output;
}

double Recognizer::distance(const cv::Mat &baseMat, const cv::Mat &targetMat) {
    cv::Mat broad;
    broad = baseMat - targetMat;
    cv::pow(broad, 2, broad);
    cv::reduce(broad, broad, 1, cv::REDUCE_SUM);

    double dis;
    cv::Point point;
    cv::minMaxLoc(broad, &dis, 0, &point, 0);
    return dis;
}

