
package com.getbouncer.cardscan.base;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;

import com.getbouncer.cardscan.base.ssd.DetectedOcrBox;
import com.getbouncer.cardscan.base.ssd.DetectionBox;
import com.getbouncer.cardscan.base.ssd.OcrPriorsGenerator;
import com.getbouncer.cardscan.base.ssd.SSD;
import com.getbouncer.cardscan.base.ssd.domain.ClassifierScores;
import com.getbouncer.cardscan.base.ssd.domain.SizeAndCenter;
import com.getbouncer.cardscan.base.util.ArrayExtensions;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

import java.util.ArrayList;
import java.util.Map;

import kotlin.jvm.functions.Function1;


public class SSDOcrDetect {
    private static SSDOcrModel ssdOcrModel = null;
    private static float[][] priors = null;

    public List<DetectedOcrBox> objectBoxes = new ArrayList<>();
    public boolean hadUnrecoverableException = false;

    /** We don't use the following two for now */
    public static boolean USE_GPU = false;

    static boolean isInit() {
        return ssdOcrModel != null;
    }

    private String ssdOutputToPredictions(Bitmap image, boolean strict){
        float[][] k_boxes = SSD.rearrangeOCRArray(ssdOcrModel.outputLocations, SSDOcrModel.FEATURE_MAP_SIZES,
                SSDOcrModel.NUM_OF_PRIORS_PER_ACTIVATION, SSDOcrModel.NUM_OF_COORDINATES);
        k_boxes = ArrayExtensions.reshape(k_boxes, SSDOcrModel.NUM_OF_COORDINATES);
        SizeAndCenter.adjustLocations(k_boxes, priors, SSDOcrModel.CENTER_VARIANCE, SSDOcrModel.SIZE_VARIANCE);
        SizeAndCenter.toRectForm(k_boxes);

        float[][] k_scores = SSD.rearrangeOCRArray(ssdOcrModel.outputClasses, SSDOcrModel.FEATURE_MAP_SIZES,
                SSDOcrModel.NUM_OF_PRIORS_PER_ACTIVATION, SSDOcrModel.NUM_OF_CLASSES);
        k_scores = ArrayExtensions.reshape(k_scores, SSDOcrModel.NUM_OF_CLASSES);
        ClassifierScores.softMax2D(k_scores);

        List<DetectionBox> detectionBoxes = SSD.extractPredictions(
            k_scores,
            k_boxes,
            new Size(image.getWidth(), image.getHeight()),
            SSDOcrModel.PROB_THRESHOLD,
            SSDOcrModel.IOU_THRESHOLD,
            SSDOcrModel.TOP_K,
            new Function1<Integer, Integer>() {
                @Override
                public Integer invoke(Integer o) {
                    if (o == 10) {
                        return 0;
                    } else {
                        return o;
                    }
                }
            }
        );

        Collections.sort(detectionBoxes, new Comparator<DetectionBox>() {
            @Override
            public int compare(DetectionBox o1, DetectionBox o2) {
                return Float.compare(o1.getRect().left, o2.getRect().left);
            }
        });

        for (DetectionBox detectionBox : detectionBoxes) {
            objectBoxes.add(new DetectedOcrBox(
                    detectionBox.getRect().left,
                    detectionBox.getRect().top,
                    detectionBox.getRect().right,
                    detectionBox.getRect().bottom,
                    detectionBox.getConfidence(),
                    detectionBox.getImageSize().getWidth(),
                    detectionBox.getImageSize().getHeight(),
                    detectionBox.getLabel()
            ));
        }

        String numberOCR;
        StringBuilder num = new StringBuilder();
        for (DetectedOcrBox box : objectBoxes) {
            num.append(box.label);
        }
        if (CreditCardUtils.isValidCardNumber(num.toString())){
            numberOCR = num.toString();
            Log.d("OCR Number passed", numberOCR);
        } else {
            Log.d("OCR Number failed", num.toString());
            if (strict) {
                numberOCR = null;
            } else {
                numberOCR = num.toString();
            }
        }

        return numberOCR;
    }

    /**
     * Run SSD Model and use the prediction API to post process
     * the model output
     */
    private String runModel(Bitmap image) {
        return runModel(image, true);
    }

    /**
     * Run SSD Model and use the prediction API to post process
     * the model output
     */
    private String runModel(Bitmap image, boolean strict) {
        final long startTime = SystemClock.uptimeMillis();

        ssdOcrModel.classifyFrame(image);
        Log.d("Before SSD Post Process", String.valueOf(SystemClock.uptimeMillis() - startTime));
        String number = ssdOutputToPredictions(image, strict);
        Log.d("After SSD Post Process", String.valueOf(SystemClock.uptimeMillis() - startTime));

        return number;
    }

    public interface OnDetectListener {
        void complete(String result);
    }

    private static <T> T getKeyByMaxValue(Map<T, ? extends Comparable> map) {
        Map.Entry<T, ? extends Comparable> maxEntry = null;
        for (Map.Entry<T, ? extends Comparable> entry : map.entrySet()) {
            if (maxEntry == null || entry.getValue().compareTo(maxEntry.getValue()) > 0) {
                maxEntry = entry;
            }
        }

        if (maxEntry != null) {
            return maxEntry.getKey();
        } else {
            return null;
        }
    }

    public static void runInCompletionLoop(
        final List<Bitmap> frames,
        final Context context,
        final OnDetectListener detection
    ) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                final Map<String, Integer> results = new HashMap<>();
                for (Bitmap ocrFrame: frames) {
                    if (ocrFrame != null) {
                        SSDOcrDetect ocrDetect = new SSDOcrDetect();
                        String prediction = ocrDetect.predict(ocrFrame, context);

                        if (prediction != null) {
                            Integer count = results.get(prediction);
                            if (count != null) {
                                results.put(prediction, count + 1);
                            } else {
                                results.put(prediction, 1);
                            }
                        }
                    }
                }

                Handler handler = new Handler(Looper.getMainLooper());
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        if (detection != null) {
                            detection.complete(getKeyByMaxValue(results));
                        }
                    }
                });
            }
        }).start();
    }

    public synchronized String predict(Bitmap image, Context context) {
        final int NUM_THREADS = 4;
        try {
            boolean createdNewModel = false;

            try{
                if (ssdOcrModel == null){
                    ssdOcrModel = new SSDOcrModel(context);
                    ssdOcrModel.setNumThreads(NUM_THREADS);
                    /** Since all the frames use the same set of priors
                     * We generate these once and use for all the frame
                     */
                    if ( priors == null){
                        priors = OcrPriorsGenerator.combinePriors();
                    }

                }
            } catch (Error | Exception e){
                Log.e("SSD", "Couldn't load ssd", e);
            }


            try {
                return runModel(image);
            } catch (Error | Exception e) {
                Log.i("OCR", "runModel exception, retry object detection", e);
                ssdOcrModel = new SSDOcrModel(context);
                return runModel(image);
            }
        } catch (Error | Exception e) {
            Log.e("OCR", "unrecoverable exception on ObjectDetect", e);
            hadUnrecoverableException = true;
            return null;
        }
    }

    public synchronized String predict(Bitmap image, Context context, boolean strict) {
        final int NUM_THREADS = 4;
        try {
            boolean createdNewModel = false;

            try{
                if (ssdOcrModel == null){
                    ssdOcrModel = new SSDOcrModel(context);
                    ssdOcrModel.setNumThreads(NUM_THREADS);
                    /** Since all the frames use the same set of priors
                     * We generate these once and use for all the frame
                     */
                    if ( priors == null){
                        priors = OcrPriorsGenerator.combinePriors();
                    }

                }
            } catch (Error | Exception e){
                Log.e("SSD", "Couldn't load ssd", e);
            }


            try {
                return runModel(image, strict);
            } catch (Error | Exception e) {
                Log.i("ObjectDetect", "runModel exception, retry object detection", e);
                ssdOcrModel = new SSDOcrModel(context);
                return runModel(image, strict);
            }
        } catch (Error | Exception e) {
            Log.e("ObjectDetect", "unrecoverable exception on ObjectDetect", e);
            hadUnrecoverableException = true;
            return null;
        }
    }

}