package com.deepfake;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;

/**
 * ImagePreprocessor - Handles all image transformation steps before passing to AI model.
 * 
 * Why Preprocessing is Needed (Viva Points):
 * 1. AI models require fixed image dimensions (e.g., 224x224 pixels).
 * 2. Models expect numerical values (floats between 0.0 and 1.0) rather than raw 0-255 bytes.
 * 3. Most vision models use NCHW tensor layout:
 *    N = Batch size (1)
 *    C = Color channels (3: Red, Green, Blue)
 *    H = Height (224)
 *    W = Width (224)
 */
public class ImagePreprocessor {

    // Standard input dimensions expected by most image classification models
    public static final int TARGET_WIDTH = 224;
    public static final int TARGET_HEIGHT = 224;
    public static final int CHANNELS = 3; // Red, Green, Blue

    /**
     * Resizes any input BufferedImage to the required model dimensions (224x224).
     * 
     * @param originalImage The uploaded user image.
     * @param targetWidth   The target width (224).
     * @param targetHeight  The target height (224).
     * @return A resized BufferedImage in RGB format.
     */
    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        // Create a new blank RGB image with the exact target dimensions
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        
        // Draw the original image onto the new resized canvas with smooth scaling
        Graphics2D graphics = resizedImage.createGraphics();
        graphics.drawImage(originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH), 0, 0, null);
        graphics.dispose();

        return resizedImage;
    }

    /**
     * Converts a BufferedImage into an ONNX input Tensor in NCHW format [1, 3, 224, 224].
     * 
     * @param image The resized RGB image.
     * @param env   The active ONNX Runtime Environment.
     * @return OnnxTensor ready for model inference.
     * @throws OrtException If tensor creation fails.
     */
    public static OnnxTensor createInputTensor(BufferedImage image, OrtEnvironment env) throws OrtException {
        // Ensure image is resized to 224x224
        BufferedImage inputImage = (image.getWidth() == TARGET_WIDTH && image.getHeight() == TARGET_HEIGHT)
                ? image
                : resizeImage(image, TARGET_WIDTH, TARGET_HEIGHT);

        // Total number of float values in the tensor: 1 * 3 * 224 * 224 = 150,528 floats
        int totalElements = 1 * CHANNELS * TARGET_WIDTH * TARGET_HEIGHT;
        FloatBuffer buffer = FloatBuffer.allocate(totalElements);

        // Separate arrays for R, G, B channel values
        float[] rChannel = new float[TARGET_WIDTH * TARGET_HEIGHT];
        float[] gChannel = new float[TARGET_WIDTH * TARGET_HEIGHT];
        float[] bChannel = new float[TARGET_WIDTH * TARGET_HEIGHT];

        int index = 0;
        for (int y = 0; y < TARGET_HEIGHT; y++) {
            for (int x = 0; x < TARGET_WIDTH; x++) {
                int rgb = inputImage.getRGB(x, y);

                // Extract individual RGB color components (0 to 255)
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // Normalize pixel values from [0, 255] to [0.0, 1.0]
                rChannel[index] = r / 255.0f;
                gChannel[index] = g / 255.0f;
                bChannel[index] = b / 255.0f;

                index++;
            }
        }

        // Put channel data into buffer in NCHW format (all Red pixels, then all Green, then all Blue)
        buffer.put(rChannel);
        buffer.put(gChannel);
        buffer.put(bChannel);
        buffer.rewind();

        // Create ONNX Tensor with shape: [Batch=1, Channels=3, Height=224, Width=224]
        long[] shape = new long[]{1, CHANNELS, TARGET_HEIGHT, TARGET_WIDTH};
        return OnnxTensor.createTensor(env, buffer, shape);
    }
}
