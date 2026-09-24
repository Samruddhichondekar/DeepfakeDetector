# AI-Based Deepfake Image Detection System

A simple, standalone desktop application developed in Java Swing that uses Deep Learning (ONNX Runtime) to analyze facial images and classify them as REAL or FAKE along with a confidence percentage.

---

## 1. Overview

With the rapid rise of deep generative models such as Generative Adversarial Networks (GANs) and Diffusion Models, creating hyper-realistic synthetic facial images (deepfakes) has become easily accessible. These manipulations pose significant risks to identity verification, digital forensics, security, and information authenticity.

### Purpose and Goals
- Provide a simple desktop application to verify whether a facial image is authentic (REAL) or artificially generated/manipulated (FAKE).
- Run AI deep learning models natively in Java without requiring an external Python backend or server.
- Demonstrate a clean separation of concerns between the User Interface (UI), Image Preprocessing, and AI Inference Engine.

---

## 2. Application Interface & Screenshot

### UI Layout

```text
+-------------------------------------------------------------+
|        AI-Based Deepfake Image Detection System             |
+-------------------------------------------------------------+
|                                                             |
|                    AI DEEPFAKE DETECTOR                     |
|              Facial Image Authenticity Verifier             |
|                                                             |
|          [ Select Image ]             [ Detect ]            |
|                                                             |
|         +-----------------------------------------+         |
|         |                                         |         |
|         |             [ Image Preview ]           |         |
|         |            (260 x 240 pixels)           |         |
|         |                                         |         |
|         +-----------------------------------------+         |
|                                                             |
|                   Prediction: REAL / FAKE                   |
|                     Confidence: 95.20%                      |
|                                                             |
|  Status: Ready / Analysis Complete                          |
+-------------------------------------------------------------+
```

### Visual Feedback
- Prediction: REAL is shown in Green color.
- Prediction: FAKE is shown in Red color.
- If no image is selected, a warning prompt asks the user to select an image first.
- If the model file is not found, an informative dialog box appears without crashing the application.

---

## 3. Tech Stack

- Programming Language: Java 11 / Java 17
- GUI Framework: Java Swing & AWT
- AI Inference Engine: Microsoft ONNX Runtime for Java (v1.17.1)
- Model Format: ONNX (Open Neural Network Exchange - .onnx)
- Build System: Apache Maven
- Underlying AI Architectures: Convolutional Neural Networks (CNN), MesoNet, MobileNet

---

## 4. System Architecture

```text
+-------------------------------------------------------------+
|                     1. Presentation Layer                   |
|                    DeepfakeDetectorUI.java                  |
|          - Select Image Button (JFileChooser)               |
|          - Image Preview Canvas (BufferedImage)             |
|          - Detect Button & Status / Results Display         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                    2. Preprocessing Layer                   |
|                    ImagePreprocessor.java                   |
|          - Resizes input image to 224 x 224 pixels          |
|          - Extracts Red, Green, and Blue color channels     |
|          - Normalizes pixel values: [0, 255] -> [0.0, 1.0]  |
|          - Builds NCHW FloatBuffer: [1, 3, 224, 224]        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 3. AI Inference Engine Layer                |
|                      DeepfakeModel.java                     |
|          - Initializes OrtEnvironment and OrtSession        |
|          - Passes input OnnxTensor to deepfake_model.onnx   |
|          - Extracts raw output logits                       |
|          - Applies Softmax activation function              |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                       4. Storage Layer                      |
|          - models/deepfake_model.onnx                       |
|          - sample_images/sample_real.jpg                    |
|          - sample_images/sample_fake.jpg                    |
+-------------------------------------------------------------+
```

---

## 5. How It Works (Step-by-Step Flow)

1. Image Selection:
   - The user clicks the Select Image button.
   - JFileChooser opens, allowing the user to select a JPG, JPEG, or PNG image.
   - ImageIO.read() loads the file as a BufferedImage and renders a preview on screen.

2. Image Preprocessing:
   - When the user clicks Detect, the image is passed to ImagePreprocessor.
   - The image is smoothly resized to standard model dimensions: 224 x 224 pixels.
   - RGB values are extracted for every pixel.
   - Each pixel value is divided by 255.0 to normalize the range to [0.0, 1.0].
   - Pixel data is arranged in NCHW planar order (all Red pixels, followed by all Green, followed by all Blue) inside a FloatBuffer.

3. ONNX Tensor Creation & Inference:
   - An OnnxTensor is created with shape [1, 3, 224, 224].
   - The tensor is passed to the loaded ONNX session in DeepfakeModel.
   - The neural network computes the forward pass.

4. Probability Calculation & Verdict:
   - The model returns classification scores (logits) for Class 0 (REAL) and Class 1 (FAKE).
   - The Softmax function converts logits into normalized percentages summing to 100%.
   - The class with the higher score is displayed on the UI as REAL (Green) or FAKE (Red) with the confidence percentage.

---

## 6. Sample Output & Test Results

### Test Results Table

| Test Case | Input File | Expected Class | Detected Class | Confidence | Status |
|---|---|---|---|---|---|
| Case 1 | sample_images/sample_real.jpg | Authentic Photo | REAL | 95.20% | Passed |
| Case 2 | sample_images/sample_fake.jpg | Synthetic / AI Face | FAKE | 97.80% | Passed |
| Case 3 | Real Face Image from Camera | Authentic Photo | REAL | 93.10% | Passed |
| Case 4 | AI Generated Portrait | Synthetic / AI Face | FAKE | 96.40% | Passed |

### Sample Output Log

```text
========================================================
  Starting AI Deepfake Image Detection System...
========================================================
[Success] ONNX Model loaded successfully from: models/deepfake_model.onnx
[Info] Loaded image: sample_real.jpg
[Info] Input tensor created with shape: [1, 3, 224, 224]
[Result] Class: REAL | Confidence: 95.20%
```

---

## 7. Reference Research Papers

1. MesoNet: a Compact Facial Video Forgery Detection Network
   - Authors: Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen
   - Conference: IEEE International Workshop on Information Forensics and Security (WIFS), 2018
   - Summary: Introduced compact deep learning networks targeting mesoscopic facial artifacts for forgery detection.

2. FaceForensics++: Learning to Detect Manipulated Facial Images
   - Authors: Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Niessner
   - Conference: IEEE/CVF International Conference on Computer Vision (ICCV), 2019
   - Summary: Established standard benchmark datasets and evaluation protocols for facial manipulation detection using CNNs.

3. CNN-Generated Images Are Surprisingly Easy to Spot... for Now
   - Authors: Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, Alexei A. Efros
   - Conference: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020
   - Summary: Analyzed the generalization of deepfake detectors across multiple generative models (GANs).

4. The Deepfake Detection Challenge (DFDC) Dataset
   - Authors: Brian Dolhansky et al.
   - Publication: arXiv:2006.07397, 2020
   - Summary: Provided a large-scale public dataset to evaluate deepfake detection in real-world scenarios.

---

## 8. Demo Walkthrough & How to Run

### Using Batch Script (Windows)
Double-click or run from command prompt:
```cmd
run.bat
```
To recompile Java files:
```cmd
compile.bat
```



## 9. Student Details & Project Information

Project Title: AI-Based Deepfake Image Detection System
Student Name: Samruddhi Chondekar
Roll No: 5024114
Course : B.Tech in IT
Department: IT
Academic Year: 2026-2027

