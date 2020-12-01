package com.wisesoft.wiseface;

public class FaceRecognizer {
    static {
        System.loadLibrary("wiseface");
    }
    public native int loadModel();
    public native String detectFaceByBase64(String base64,int type);
    public native String detectMaskByBase64(String base64,int type);
    public native String extractFaceFeatureByBase64(String base64,int detected,int type);
    public native String computeDistance(String base_emb, String target_emb);
    public native String computeDistanceByBase64(String base_data,String target_data, int detected);
}
