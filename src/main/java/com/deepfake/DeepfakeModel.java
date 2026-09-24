package com.deepfake;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Collections;

/**
 * DeepfakeModel - Manages ONNX model loading and AI inference.
 * 
 * Core Responsibilities:
 * 1. Initializes the ONNX Runtime environment and session.
 * 2. Feeds preprocessed image tensors into the model.
 * 3. Reads model prediction output (probabilities/logits).
 * 4. Determines whether the image is "REAL" or "FAKE" with a confidence score.
 */
public class DeepfakeModel {

    // Default path to the ONNX model file
    private static final String DEFAULT_MODEL_PATH = "models/deepfake_model.onnx";

    private OrtEnvironment env;
    private OrtSession session;
    private boolean isLoaded = false;
    private String modelPath;

    /**
     * PredictionResult holds the classification outcome.
     */
    public static class PredictionResult {
        private final String label;          // "REAL" or "FAKE"
        private final float confidence;      // e.g., 94.5 (percentage)
        private final float realScore;       // Probability/score for REAL
        private final float fakeScore;       // Probability/score for FAKE

        public PredictionResult(String label, float confidence, float realScore, float fakeScore) {
            this.label = label;
            this.confidence = confidence;
            this.realScore = realScore;
            this.fakeScore = fakeScore;
        }

        public String getLabel() {
            return label;
        }

        public float getConfidence() {
            return confidence;
        }

        public float getRealScore() {
            return realScore;
        }

        public float getFakeScore() {
            return fakeScore;
        }
    }

    /**
     * Constructor - Attempts to load the ONNX model from the default models directory.
     */
    public DeepfakeModel() {
        this(DEFAULT_MODEL_PATH);
    }

    /**
     * Constructor with custom model file path.
     */
    public DeepfakeModel(String modelPath) {
        this.modelPath = modelPath;
        loadModel();
    }

    /**
     * Loads the ONNX model file if present.
     * Safe loading: If the file is not found, it marks isLoaded as false without crashing.
     */
    public boolean loadModel() {
        File file = new File(modelPath);
        if (!file.exists() || !file.isFile()) {
            this.isLoaded = false;
            System.out.println("[Notice] ONNX model file not found at: " + file.getAbsolutePath());
            return false;
        }

        try {
            // Create ONNX Runtime environment and inference session
            this.env = OrtEnvironment.getEnvironment();
            this.session = env.createSession(file.getAbsolutePath(), new OrtSession.SessionOptions());
            this.isLoaded = true;
            System.out.println("[Success] ONNX Model loaded successfully from: " + file.getAbsolutePath());
            return true;
        } catch (Exception e) {
            this.isLoaded = false;
            System.err.println("[Error] Failed to load ONNX model: " + e.getMessage());
            return false;
        }
    }

    /**
     * Checks whether the AI model is ready for inference.
     */
    public boolean isLoaded() {
        return isLoaded && session != null;
    }

    /**
     * Runs AI inference on a preprocessed input image.
     * 
     * @param image The input BufferedImage.
     * @return PredictionResult containing REAL/FAKE label and confidence.
     * @throws Exception If model is not loaded or inference fails.
     */
    public PredictionResult predict(BufferedImage image) throws Exception {
        if (!isLoaded()) {
            throw new IllegalStateException("AI model not found.\nPlease place deepfake_model.onnx inside the models folder.");
        }

        // Step 1: Preprocess image into ONNX Tensor [1, 3, 224, 224]
        try (OnnxTensor inputTensor = ImagePreprocessor.createInputTensor(image, env)) {
            
            // Get the input name from the model's metadata
            String inputName = session.getInputNames().iterator().next();

            // Step 2: Run inference in ONNX Runtime
            try (OrtSession.Result results = session.run(Collections.singletonMap(inputName, inputTensor))) {
                
                // Step 3: Extract output values
                OnnxValue outputValue = results.get(0);
                float[] probabilities = extractProbabilities(outputValue);

                // probabilities[0] = REAL score, probabilities[1] = FAKE score
                float realProb = probabilities[0];
                float fakeProb = probabilities[1];

                // Step 4: Determine classification decision
                String label;
                float confidence;

                if (fakeProb > realProb) {
                    label = "FAKE";
                    confidence = fakeProb * 100.0f;
                } else {
                    label = "REAL";
                    confidence = realProb * 100.0f;
                }

                return new PredictionResult(label, confidence, realProb, fakeProb);
            }
        }
    }

    /**
     * Helper to extract and normalize probabilities from model output.
     * Handles 2-class logits [1, 2] (via Softmax) or 1-value binary output [1, 1].
     */
    private float[] extractProbabilities(OnnxValue outputValue) throws OrtException {
        Object rawValue = outputValue.getValue();

        // Case A: Model outputs 2D array [batch, classes] e.g. [[real_logit, fake_logit]]
        if (rawValue instanceof float[][]) {
            float[][] array = (float[][]) rawValue;
            float[] logits = array[0];

            if (logits.length >= 2) {
                return applySoftmax(logits[0], logits[1]);
            } else if (logits.length == 1) {
                // Single sigmoid value representing FAKE probability
                float pFake = sigmoid(logits[0]);
                float pReal = 1.0f - pFake;
                return new float[]{pReal, pFake};
            }
        }
        
        // Case B: Model outputs 1D array [classes]
        if (rawValue instanceof float[]) {
            float[] logits = (float[]) rawValue;
            if (logits.length >= 2) {
                return applySoftmax(logits[0], logits[1]);
            }
        }

        // Fallback default
        return new float[]{0.5f, 0.5f};
    }

    /**
     * Softmax function converts raw logits into normalized probabilities that sum to 1.0.
     * P(i) = exp(z_i) / (exp(z_1) + exp(z_2))
     */
    private float[] applySoftmax(float z1, float z2) {
        float max = Math.max(z1, z2); // For numerical stability
        float exp1 = (float) Math.exp(z1 - max);
        float exp2 = (float) Math.exp(z2 - max);
        float sum = exp1 + exp2;
        return new float[]{exp1 / sum, exp2 / sum};
    }

    /**
     * Sigmoid activation function for single-output binary models: 1 / (1 + exp(-x))
     */
    private float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    /**
     * Closes the ONNX session and environment to free native memory.
     */
    public void close() {
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        } catch (Exception e) {
            // Ignore close errors
        }
    }
}
