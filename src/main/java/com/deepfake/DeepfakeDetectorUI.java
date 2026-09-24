package com.deepfake;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * DeepfakeDetectorUI - Simple Swing Desktop Interface for AI Deepfake Detection.
 * 
 * Flow:
 * 1. User clicks "Select Image" -> Chooses JPG/JPEG/PNG image via JFileChooser.
 * 2. Selected image is displayed in the Image Preview box.
 * 3. User clicks "Detect" -> Passes image to DeepfakeModel for ONNX inference.
 * 4. Displays Prediction (REAL / FAKE) and Confidence percentage.
 */
public class DeepfakeDetectorUI extends JFrame {

    // UI Components
    private JLabel titleLabel;
    private JButton selectButton;
    private JButton detectButton;
    private JLabel imagePreviewLabel;
    private JLabel predictionLabel;
    private JLabel confidenceLabel;
    private JLabel statusLabel;

    // Loaded image and AI model reference
    private BufferedImage selectedImage;
    private File selectedImageFile;
    private DeepfakeModel model;

    public DeepfakeDetectorUI() {
        // Initialize AI model
        this.model = new DeepfakeModel();

        // Setup Main Window
        initWindow();

        // Build User Interface
        initUI();
    }

    /**
     * Configures the main JFrame properties.
     */
    private void initWindow() {
        setTitle("AI-Based Deepfake Image Detection System");
        setSize(480, 620);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null); // Center window on screen
        setResizable(false);
    }

    /**
     * Builds and arranges all Swing UI components.
     */
    private void initUI() {
        // Main Container with padding
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.setBorder(new EmptyBorder(20, 25, 20, 25));
        mainPanel.setBackground(new Color(245, 247, 250));

        // 1. Header Title
        titleLabel = new JLabel("AI DEEPFAKE DETECTOR", SwingConstants.CENTER);
        titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 20));
        titleLabel.setForeground(new Color(33, 37, 41));
        titleLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        JLabel subtitleLabel = new JLabel("Facial Image Authenticity Verifier", SwingConstants.CENTER);
        subtitleLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        subtitleLabel.setForeground(new Color(108, 117, 125));
        subtitleLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        // 2. Buttons Panel: [ Select Image ] and [ Detect ]
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 15, 10));
        buttonPanel.setOpaque(false);

        selectButton = new JButton("Select Image");
        selectButton.setFont(new Font("Segoe UI", Font.BOLD, 13));
        selectButton.setFocusPainted(false);
        selectButton.setPreferredSize(new Dimension(140, 36));
        selectButton.addActionListener(e -> onSelectImage());

        detectButton = new JButton("Detect");
        detectButton.setFont(new Font("Segoe UI", Font.BOLD, 13));
        detectButton.setFocusPainted(false);
        detectButton.setPreferredSize(new Dimension(140, 36));
        detectButton.addActionListener(e -> onDetect());

        buttonPanel.add(selectButton);
        buttonPanel.add(detectButton);

        // 3. Image Preview Area
        imagePreviewLabel = new JLabel("No Image Selected", SwingConstants.CENTER);
        imagePreviewLabel.setFont(new Font("Segoe UI", Font.ITALIC, 13));
        imagePreviewLabel.setForeground(new Color(140, 145, 150));
        imagePreviewLabel.setPreferredSize(new Dimension(280, 260));
        imagePreviewLabel.setMaximumSize(new Dimension(280, 260));
        imagePreviewLabel.setBorder(new LineBorder(new Color(200, 205, 210), 1, true));
        imagePreviewLabel.setBackground(Color.WHITE);
        imagePreviewLabel.setOpaque(true);
        imagePreviewLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        // 4. Results Panel: Prediction and Confidence
        JPanel resultPanel = new JPanel();
        resultPanel.setLayout(new BoxLayout(resultPanel, BoxLayout.Y_AXIS));
        resultPanel.setOpaque(false);
        resultPanel.setAlignmentX(Component.CENTER_ALIGNMENT);

        predictionLabel = new JLabel("Prediction: ---", SwingConstants.CENTER);
        predictionLabel.setFont(new Font("Segoe UI", Font.BOLD, 18));
        predictionLabel.setForeground(new Color(50, 50, 50));
        predictionLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        confidenceLabel = new JLabel("Confidence: ---", SwingConstants.CENTER);
        confidenceLabel.setFont(new Font("Segoe UI", Font.PLAIN, 15));
        confidenceLabel.setForeground(new Color(80, 80, 80));
        confidenceLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        resultPanel.add(predictionLabel);
        resultPanel.add(Box.createRigidArea(new Dimension(0, 6)));
        resultPanel.add(confidenceLabel);

        // 5. Status / Footer Label
        statusLabel = new JLabel("Status: Ready", SwingConstants.CENTER);
        statusLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        statusLabel.setForeground(new Color(120, 120, 120));
        statusLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        // Assemble all components into main panel with vertical spacing
        mainPanel.add(titleLabel);
        mainPanel.add(Box.createRigidArea(new Dimension(0, 4)));
        mainPanel.add(subtitleLabel);
        mainPanel.add(Box.createRigidArea(new Dimension(0, 15)));
        mainPanel.add(buttonPanel);
        mainPanel.add(Box.createRigidArea(new Dimension(0, 15)));
        mainPanel.add(imagePreviewLabel);
        mainPanel.add(Box.createRigidArea(new Dimension(0, 18)));
        mainPanel.add(resultPanel);
        mainPanel.add(Box.createVerticalGlue());
        mainPanel.add(statusLabel);

        add(mainPanel);
    }

    /**
     * Action for [ Select Image ] button.
     * Opens a JFileChooser to select a JPG/PNG image and previews it.
     */
    private void onSelectImage() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Select Facial Image");
        
        // Filter for image files only
        FileNameExtensionFilter filter = new FileNameExtensionFilter("Image Files (*.jpg, *.jpeg, *.png)", "jpg", "jpeg", "png");
        fileChooser.setFileFilter(filter);

        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            try {
                BufferedImage img = ImageIO.read(file);
                if (img == null) {
                    JOptionPane.showMessageDialog(this, "The selected file is not a valid image.", "Invalid Image", JOptionPane.ERROR_MESSAGE);
                    return;
                }

                this.selectedImage = img;
                this.selectedImageFile = file;

                // Scale image for preview
                Image scaled = img.getScaledInstance(260, 240, Image.SCALE_SMOOTH);
                imagePreviewLabel.setText("");
                imagePreviewLabel.setIcon(new ImageIcon(scaled));

                // Reset previous results
                predictionLabel.setText("Prediction: ---");
                predictionLabel.setForeground(new Color(50, 50, 50));
                confidenceLabel.setText("Confidence: ---");
                statusLabel.setText("Loaded: " + file.getName());

            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Failed to load image: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    /**
     * Action for [ Detect ] button.
     * Passes the image through ImagePreprocessor and DeepfakeModel to display REAL / FAKE.
     */
    private void onDetect() {
        // Step 1: Validate if an image has been selected
        if (selectedImage == null) {
            JOptionPane.showMessageDialog(this, "Please select an image first.", "No Image Selected", JOptionPane.WARNING_MESSAGE);
            return;
        }

        // Step 2: Validate if AI model is loaded
        if (!model.isLoaded()) {
            // Re-attempt loading in case model was added after app start
            boolean reloaded = model.loadModel();
            if (!reloaded) {
                JOptionPane.showMessageDialog(
                        this,
                        "AI model not found.\nPlease place deepfake_model.onnx inside the models folder.",
                        "Model Missing",
                        JOptionPane.ERROR_MESSAGE
                );
                return;
            }
        }

        // Step 3: Run AI prediction
        try {
            statusLabel.setText("Analyzing image...");
            
            DeepfakeModel.PredictionResult result = model.predict(selectedImage);

            // Step 4: Display Prediction and Confidence
            String label = result.getLabel(); // "REAL" or "FAKE"
            float conf = result.getConfidence();

            predictionLabel.setText("Prediction: " + label);
            confidenceLabel.setText(String.format("Confidence: %.2f%%", conf));

            // Visual feedback: Green for REAL, Red for FAKE
            if ("REAL".equalsIgnoreCase(label)) {
                predictionLabel.setForeground(new Color(25, 135, 84)); // Green
            } else {
                predictionLabel.setForeground(new Color(220, 53, 69)); // Red
            }

            statusLabel.setText("Analysis Complete");

        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "Prediction Error:\n" + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            statusLabel.setText("Error during detection");
        }
    }
}
