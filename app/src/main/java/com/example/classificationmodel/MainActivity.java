package com.example.classificationmodel;

import androidx.appcompat.app.AppCompatActivity;
import android.Manifest;

import androidx.annotation.Nullable;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.Button;
import android.widget.TextView;
import android.graphics.Bitmap;
import android.net.Uri;
import android.content.Context;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import java.io.IOException;



public class MainActivity extends AppCompatActivity {
    private static final int numClasses = 3;
    private Interpreter interpreter;

    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize=32;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera=findViewById(R.id.button);
        gallery=findViewById(R.id.button2);

        result=findViewById(R.id.result);
        imageView=findViewById(R.id.imageView);


        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view){
                if(checkSelfPermission(Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                }else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view){
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 3);

            }
        });
    }

//    public void classify(Bitmap image){
//        TensorBuffer outputFeature0=null;
//        try {
//
//
//            Interpreter.Options options=new Interpreter.Options();
//            Interpreter interpreter=new Interpreter(FileUtil.loadMappedFile(getApplicationContext(), "model1.tflite"));
//
//            // Creates inputs for reference.
//            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
//            ByteBuffer byteBuffer=ByteBuffer.allocateDirect(4*imageSize*imageSize*3);
//            byteBuffer.order(ByteOrder.nativeOrder());
//
//            int[] intValues = new int[imageSize * imageSize];
//            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
//            int pixel = 0;
//            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
//            for(int i = 0; i < imageSize; i ++){
//                for(int j = 0; j < imageSize; j++){
//                    int val = intValues[pixel++]; // RGB
//                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
//                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
//                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
//                }
//            }
//
//            inputFeature0.loadBuffer(byteBuffer);
//
//            // Runs model inference and gets result.
//            //Interpreter.Outputs outputs = interpreter.process(inputFeature0);
//
//            //TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//
//            interpreter.run(inputFeature0.getBuffer(), outputFeature0.getBuffer());
//            float[] confidences=outputFeature0.getFloatArray();
//            int maxPos=0;
//            float maxConfidence=0;
//            for (int i=0; i <confidences.length; i++){
//                if (confidences[i]>maxConfidence){
//                    maxConfidence=confidences[i];
//                    maxPos=i;
//                }
//            }
//            String[] classes={"Bird","Cat","Fish"};
//            result.setText(classes[maxPos]);
//            // Releases model resources if no longer used.
//            interpreter.close();
//        } catch (IOException e) {
//            // TODO Handle the exception
//        }
//    }
    public void classify(Bitmap image) {
        try {
            Interpreter.Options options = new Interpreter.Options();
            Interpreter interpreter = new Interpreter(FileUtil.loadMappedFile(getApplicationContext(), "model1.tflite"));

            // Ensure the input image is resized to match the model's input size (32x32).
            Bitmap resizedImage = Bitmap.createScaledBitmap(image, 32, 32, true);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 32 * 32 * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[32 * 32];
            resizedImage.getPixels(intValues, 0, resizedImage.getWidth(), 0, 0, resizedImage.getWidth(), resizedImage.getHeight());
            int pixel = 0;

            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(new int[]{1, numClasses}, DataType.FLOAT32);

            interpreter.run(inputFeature0.getBuffer(), outputFeature0.getBuffer());

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Bird", "Cat", "Fish"};
            result.setText(classes[maxPos]);
        } catch (IOException e) {
            // Handle the exception, e.g., log it or show an error message.
        } finally {
            // Close the interpreter to release resources.
            if (interpreter != null) {
                interpreter.close();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data){
        if(resultCode==RESULT_OK){
            if(requestCode==3){
                Bitmap image=(Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image= ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classify(image);

            }else{
                Uri dat =data.getData();
                Bitmap image=null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e){
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                //classify(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}