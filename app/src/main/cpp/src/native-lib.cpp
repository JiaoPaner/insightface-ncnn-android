#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "recognizer.h"
#include "ncnn/net.h"
#include <android/log.h>
#include "utils.h"
#include "cJSON.h"
#include "detector.h"

static Recognizer recognizer;
static Detector detector;

char* extractFaceFeatureByFace(cv::Mat &face){
    resize(face, face, cv::Size(112, 112));
    cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
    char *resultJson;
    try {
        if (face.empty()) {
            cJSON_AddNumberToObject(result, "status", -1);
            cJSON_AddStringToObject(result, "msg", "register failed,there is no face");
            cJSON_AddItemToObject(result, "embeddings", embeddings);
            resultJson = cJSON_PrintUnformatted(result);
            return resultJson;
        }

        cv::Mat features = recognizer.extractFeature(face);
        std::vector<double> vector = (std::vector<double>)features;

        std::stringstream ss;
        ss << std::setprecision(16);
        std::copy(vector.begin(), vector.end(), std::ostream_iterator<double>(ss, ","));
        std::string values = ss.str();
        values.pop_back();

        cJSON  *embedding;
        cJSON_AddItemToArray(embeddings, embedding = cJSON_CreateObject());
        cJSON_AddStringToObject(embedding, "embedding", values.c_str());

        cJSON_AddNumberToObject(result, "status", 1);
        cJSON_AddStringToObject(result, "msg", "register success");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "register failed");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
}

char* extractFaceFeatureByImage(cv::Mat image,int type) {
    cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
    char *resultJson;
    try {
        std::vector<cv::Mat>  aligned_faces = detector.createAlignFace(image,type);
        if (aligned_faces.size() == 0) {
            cJSON_AddNumberToObject(result, "status", -1);
            cJSON_AddStringToObject(result, "msg", "there is no face");
            cJSON_AddItemToObject(result, "embeddings", embeddings);
            resultJson = cJSON_PrintUnformatted(result);
            return resultJson;
        }
        for (int i = 0; i < aligned_faces.size(); i++) {
            cv::Mat features = recognizer.extractFeature(aligned_faces[i]);
            std::vector<double> vector = (std::vector<double>)features;

            std::stringstream ss;
            ss << std::setprecision(16);
            std::copy(vector.begin(), vector.end(), std::ostream_iterator<double>(ss, ","));
            std::string values = ss.str();
            values.pop_back();

            cJSON  *embedding;
            cJSON_AddItemToArray(embeddings, embedding = cJSON_CreateObject());
            cJSON_AddStringToObject(embedding, "embedding", values.c_str());
        }
        cJSON_AddNumberToObject(result, "status", 1);
        cJSON_AddStringToObject(result, "msg", "register success");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "register failed");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
}

char* computeDistanceByMat(cv::Mat& base, cv::Mat& target,int detected) {
    cJSON  *result = cJSON_CreateObject();
    char *resultJson;
    double distance,sim = 0;
    Mat base_emb, target_emb;
    try {

        if (detected == 1) {
            base_emb = recognizer.extractFeature(base);
            target_emb = recognizer.extractFeature(target);
            distance = recognizer.distance(base_emb, target_emb);
        }
        else {
            std::vector<cv::Mat> base_vector = detector.createAlignFace(base,1);
            std::vector<cv::Mat> target_vector = detector.createAlignFace(target,1);

            if ((base_vector.empty() || target_vector.empty())) {
                cJSON_AddNumberToObject(result, "status", -1);
                cJSON_AddStringToObject(result, "msg", "compute failed,one of images has no face");
                cJSON_AddNumberToObject(result, "distance", -1);
                cJSON_AddNumberToObject(result, "sim", 0);
                resultJson = cJSON_PrintUnformatted(result);
                return resultJson;
            }
            base_emb = recognizer.extractFeature(base_vector[0]);
            target_emb = recognizer.extractFeature(target_vector[0]);
            distance = recognizer.distance(base_emb, target_emb);
        }

        sim = base_emb.dot(target_emb);
        if (sim < 0)
            sim = 0;
        if (sim > 100)
            sim = 100;

        cJSON_AddNumberToObject(result, "status", 1);
        cJSON_AddStringToObject(result, "msg", "compute success");
        cJSON_AddNumberToObject(result, "distance",distance);
        cJSON_AddNumberToObject(result, "sim", sim * 100);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -2);
        cJSON_AddStringToObject(result, "msg", "compute failed");
        cJSON_AddNumberToObject(result, "distance", -1);
        cJSON_AddNumberToObject(result, "sim", 0);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
}

cv::Mat convertToMat(std::string str) {
    std::vector<double> v;
    std::stringstream ss(str);
    ss << std::setprecision(16);
    std::string token;
    while (std::getline(ss, token, ',')) {
        v.push_back(std::stod(token));
    }
    cv::Mat output = cv::Mat(v, true).reshape(1, 1);
    return output;
}

/*----------------------------------------------------------api list-----------------------------------------------------------------------------*/
/*
	detected = 0:normal image file that includes faces
	detected = 1:face image that only includes single face

	type = 0:all faces
	type = 1:max face

    distance < 1:same person or not

*/
/**
 * 加载模型
 */
extern "C"
JNIEXPORT jint JNICALL
Java_com_wisesoft_wiseface_FaceRecognizer_loadModel(JNIEnv *env, jobject thiz) {
    int  reconize = recognizer.loadModel();//识别模型
    int  detect = detector.loadModel();//人脸检测模型
    if(reconize == 1 && detect == 1){
        return jint(1);
    }

    return jint (0);
}
/**
 * 提取人脸特征
 */
extern "C"
JNIEXPORT jstring JNICALL
Java_com_wisesoft_wiseface_FaceRecognizer_extractFaceFeatureByBase64(JNIEnv *env, jobject thiz,
                                                                   jstring base64, jint detected,
                                                                   jint type) {
    const char* str = env->GetStringUTFChars(base64, JNI_FALSE);

    std::string data(str);
    cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
    char *resultJson;
    cv::Mat image;
    try {
        image = Utils::base64ToMat(data);
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "register failed,can not convert base64 to Mat");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return env->NewStringUTF(resultJson);
    }
    if (detected == 1) {
        resultJson = extractFaceFeatureByFace(image);
    }
    else {
        resultJson = extractFaceFeatureByImage(image,type);
    }
    return env->NewStringUTF(resultJson);
}
/**
 * 相似度计算
 */
extern "C"
JNIEXPORT jstring JNICALL
Java_com_wisesoft_wiseface_FaceRecognizer_computeDistance(JNIEnv *env, jobject thiz, jstring base_emb,
                                                        jstring target_emb) {
    cJSON  *result = cJSON_CreateObject();
    char *resultJson;
    const char* base_emb_str = env->GetStringUTFChars(base_emb, JNI_FALSE);
    const char* target_emb_str = env->GetStringUTFChars(target_emb, JNI_FALSE);
    try{
        std::string base(base_emb_str), target(target_emb_str);
        cv::Mat baseMat = convertToMat(base), targetMat = convertToMat(target);
        double distance = recognizer.distance(baseMat, targetMat);

        double sim = baseMat.dot(targetMat);
        if (sim < 0)
            sim = 0;
        if (sim > 100)
            sim = 100;
        cJSON_AddNumberToObject(result, "status", 1);
        cJSON_AddStringToObject(result, "msg", "compute success");
        cJSON_AddNumberToObject(result, "distance", distance);
        cJSON_AddNumberToObject(result, "sim", sim * 100);
        resultJson = cJSON_PrintUnformatted(result);
        return env->NewStringUTF(resultJson);
    }
    catch (const std::exception&){
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "compute failed");
        cJSON_AddNumberToObject(result, "distance", -1);
        cJSON_AddNumberToObject(result, "sim", 0);
        resultJson = cJSON_PrintUnformatted(result);
        return env->NewStringUTF(resultJson);
    }
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_wisesoft_wiseface_FaceRecognizer_computeDistanceByBase64(JNIEnv *env, jobject thiz,
                                                                jstring base_data,
                                                                jstring target_data,
                                                                jint detected) {
    const char* base_emb_str = env->GetStringUTFChars(base_data, JNI_FALSE);
    const char* target_emb_str = env->GetStringUTFChars(target_data, JNI_FALSE);
    std::string base_str(base_emb_str);
    std::string target_str(target_emb_str);
    cJSON  *result = cJSON_CreateObject();
    char *resultJson;
    cv::Mat base, target;
    try {
        base = Utils::base64ToMat(base_str);
        target = Utils::base64ToMat(target_str);
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "can not convert base64");
        cJSON_AddNumberToObject(result, "distance", -1);
        cJSON_AddNumberToObject(result, "sim", 0);
        resultJson = cJSON_PrintUnformatted(result);
        return env->NewStringUTF(resultJson);
    }
    resultJson = computeDistanceByMat(base, target, detected);
    return env->NewStringUTF(resultJson);
}