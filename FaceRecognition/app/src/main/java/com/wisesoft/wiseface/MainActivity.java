package com.wisesoft.wiseface;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("wiseface");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        TextView tv = findViewById(R.id.sample_text);
        loadModel();
        Thread thread=new Thread(new Runnable() {
            @Override
            public void run() {
                    Bitmap img1 = null;
                    try {
                        InputStream is = getAssets().open("test2.png");
                        img1 = BitmapFactory.decodeStream(is);
                        is.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    //Bitmap image = img.copy(Bitmap.Config.ARGB_8888, true);
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    //读取图片到ByteArrayOutputStream
                    img1.compress(Bitmap.CompressFormat.PNG, 40, baos); //参数如果为100那么就不压缩
                    byte[] bytes = baos.toByteArray();
                    String strbm = Base64.encodeToString(bytes,Base64.DEFAULT);

                    Bitmap img2 = null;
                    try {
                        InputStream is2 = getAssets().open("test3.png");
                        img2 = BitmapFactory.decodeStream(is2);
                        is2.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    //Bitmap image = img.copy(Bitmap.Config.ARGB_8888, true);
                    ByteArrayOutputStream baos2 = new ByteArrayOutputStream();
                    //读取图片到ByteArrayOutputStream
                    img2.compress(Bitmap.CompressFormat.PNG, 40, baos2); //参数如果为100那么就不压缩
                    byte[] bytes2 = baos2.toByteArray();
                    String strbm2 = Base64.encodeToString(bytes2,Base64.DEFAULT);

                    //System.out.println("img1:"+strbm);
                    //System.out.println("img2:"+strbm2);

                    long start = System.currentTimeMillis();
                    String result = computeDistanceByBase64(strbm,strbm2,0);
                    long end = System.currentTimeMillis();
                    long interval = end - start;
                    System.out.println(interval+"");
                    System.out.println(result);
            }
        });
        thread.start();
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native int loadModel();
    public native String extractFaceFeatureByBase64(String base64,int detected,int type);
    public native String computeDistance(String base_emb, String target_emb);
    public native String computeDistanceByBase64(String base_data,String target_data, int detected);

}
