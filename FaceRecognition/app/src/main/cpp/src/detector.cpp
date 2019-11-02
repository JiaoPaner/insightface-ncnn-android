//
// Created by jiaopan on 2019-11-02.
//

#include "detector.h"
#include "retina.id.h"
#include "retina.mem.h"
#include "face_align.hpp"

int Detector::loadModel() {
    int param = this->net.load_param(retina_param_bin);
    int model = this->net.load_model(retina_bin);
    if(param > 0 && model > 0){
        return 1;
    }
    return 0;
}

std::vector<cv::Mat> Detector::createAlignFace(cv::Mat &img, int type) {
    std::vector<cv::Mat> aligned_faces;
    if (img.empty())
        return aligned_faces;
    cv::Mat src(5, 2, CV_32FC1, norm_face);
    std::vector<FaceObject> face_boxs;
    this->detect_retinaface(img,face_boxs);
    int index = 0;
    float max_box = 0;

    if (type == 1) {
        if (face_boxs.size() > 0) {
            for (int i = 0; i < face_boxs.size(); i++) {
                FaceObject face_box = face_boxs[i];
                float box = (face_box.rect.width + face_box.rect.height) / 2;
                if (box > max_box) {
                    max_box = box;
                    index = i;
                }
            }
            FaceObject face_box = face_boxs[index];
            //cv::rectangle(img, cv::Point(face_box.x0, face_box.y0), cv::Point(face_box.x1, face_box.y1), cv::Scalar(0, 255, 0), 2);
            //cv::imshow("img",img);
            //cv::waitKey(2000);

            float landmark[5][2] = {
                    { face_box.landmark[0].x , face_box.landmark[0].y },
                    { face_box.landmark[1].x , face_box.landmark[1].y },
                    { face_box.landmark[2].x , face_box.landmark[2].y },
                    { face_box.landmark[3].x , face_box.landmark[3].y },
                    { face_box.landmark[4].x , face_box.landmark[4].y }
            };

            cv::Mat dst(5, 2, CV_32FC1, landmark);
            cv::Mat m = similarTransform(dst, src);
            cv::Mat aligned(112, 112, CV_32FC3);
            cv::Size size(112, 112);
            cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
            cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
            aligned_faces.push_back(aligned);
        }
    }
    else{
        for (int i = 0; i < face_boxs.size(); i++) {
            FaceObject face_box = face_boxs[i];
            if (face_box.prob * 100 < 70)
                continue;
            float landmark[5][2] = {
                    { face_box.landmark[0].x , face_box.landmark[0].y },
                    { face_box.landmark[1].x , face_box.landmark[1].y },
                    { face_box.landmark[2].x , face_box.landmark[2].y },
                    { face_box.landmark[3].x , face_box.landmark[3].y },
                    { face_box.landmark[4].x , face_box.landmark[4].y }
            };
            cv::Mat dst(5, 2, CV_32FC1, landmark);
            cv::Mat m = similarTransform(dst, src);
            cv::Mat aligned(112, 112, CV_32FC3);
            cv::Size size(112, 112);
            cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
            cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
            aligned_faces.push_back(aligned);
        }
    }
    return aligned_faces;
}

int Detector::detect_retinaface(const cv::Mat &bgr, std::vector<FaceObject> &faceobjects) {
    const float prob_threshold = 0.8f;
    const float nms_threshold = 0.4f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);

    ncnn::Extractor ex = this->net.create_extractor();

    ex.input(retina_param_id::BLOB_data, in);

    std::vector<FaceObject> faceproposals;
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract(retina_param_id::BLOB_face_rpn_cls_prob_reshape_stride32, score_blob);
        ex.extract(retina_param_id::BLOB_face_rpn_bbox_pred_stride32, bbox_blob);
        ex.extract(retina_param_id::BLOB_face_rpn_landmark_pred_stride32, landmark_blob);

        const int base_size = 16;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 32.f;
        scales[1] = 16.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract(retina_param_id::BLOB_face_rpn_cls_prob_reshape_stride16, score_blob);
        ex.extract(retina_param_id::BLOB_face_rpn_bbox_pred_stride16, bbox_blob);
        ex.extract(retina_param_id::BLOB_face_rpn_landmark_pred_stride16, landmark_blob);

        const int base_size = 16;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract(retina_param_id::BLOB_face_rpn_cls_prob_reshape_stride8, score_blob);
        ex.extract(retina_param_id::BLOB_face_rpn_bbox_pred_stride8, bbox_blob);
        ex.extract(retina_param_id::BLOB_face_rpn_landmark_pred_stride8, landmark_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 2.f;
        scales[1] = 1.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();

    faceobjects.resize(face_count);
    for (int i = 0; i < face_count; i++){
        faceobjects[i] = faceproposals[ picked[i] ];

        // clip to image size
        float x0 = faceobjects[i].rect.x;
        float y0 = faceobjects[i].rect.y;
        float x1 = x0 + faceobjects[i].rect.width;
        float y1 = y0 + faceobjects[i].rect.height;

        x0 = std::max(std::min(x0, (float)img_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float)img_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float)img_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float)img_h - 1), 0.f);

        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;
    }

    return 0;
}

ncnn::Mat Detector::generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales) {
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++){
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar);//round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++){
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

void Detector::generate_proposals(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob,
                                  const ncnn::Mat &bbox_blob, const ncnn::Mat &landmark_blob,
                                  float prob_threshold, std::vector<FaceObject> &faceobjects) {
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q=0; q<num_anchors; q++){
        const float* anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q + num_anchors);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i=0; i<h; i++){
            float anchor_x = anchor[0];
            for (int j=0; j<w; j++){
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold){
                    // apply center size
                    float dx = bbox.channel(0)[index];
                    float dy = bbox.channel(1)[index];
                    float dw = bbox.channel(2)[index];
                    float dh = bbox.channel(3)[index];

                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float pb_cx = cx + anchor_w * dx;
                    float pb_cy = cy + anchor_h * dy;

                    float pb_w = anchor_w * exp(dw);
                    float pb_h = anchor_h * exp(dh);

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.landmark[0].x = cx + (anchor_w + 1) * landmark.channel(0)[index];
                    obj.landmark[0].y = cy + (anchor_h + 1) * landmark.channel(1)[index];
                    obj.landmark[1].x = cx + (anchor_w + 1) * landmark.channel(2)[index];
                    obj.landmark[1].y = cy + (anchor_h + 1) * landmark.channel(3)[index];
                    obj.landmark[2].x = cx + (anchor_w + 1) * landmark.channel(4)[index];
                    obj.landmark[2].y = cy + (anchor_h + 1) * landmark.channel(5)[index];
                    obj.landmark[3].x = cx + (anchor_w + 1) * landmark.channel(6)[index];
                    obj.landmark[3].y = cy + (anchor_h + 1) * landmark.channel(7)[index];
                    obj.landmark[4].x = cx + (anchor_w + 1) * landmark.channel(8)[index];
                    obj.landmark[4].y = cy + (anchor_h + 1) * landmark.channel(9)[index];
                    obj.prob = prob;

                    faceobjects.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
}

void Detector::qsort_descent_inplace(std::vector<FaceObject> &faceobjects) {
    if (faceobjects.empty())
        return;
    this->qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void Detector::qsort_descent_inplace(std::vector<FaceObject> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j){
        while (faceobjects[i].prob > p)
            i++;
        while (faceobjects[j].prob < p)
            j--;
        if (i <= j){
            // swap
            //std::swap(faceobjects[i], faceobjects[j]);
            FaceObject faceObject = faceobjects.at(i);
            faceobjects.at(i) = faceobjects.at(j);
            faceobjects.at(j) = faceObject;
            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void Detector::nms_sorted_bboxes(const std::vector<FaceObject> &faceobjects, std::vector<int> &picked,float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++){
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++){
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++){
            const FaceObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
//             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

float Detector::intersection_area(const FaceObject &a, const FaceObject &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}




