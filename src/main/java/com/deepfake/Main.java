package com.deepfake;

import javax.swing.SwingUtilities;
import javax.swing.UIManager;

/**
 * Main Class - The entry point of the AI Deepfake Image Detection application.
 * 
 * Simple flow:
 * 1. Sets the system Look and Feel for clean native UI.
 * 2. Initializes and displays the DeepfakeDetectorUI on the Swing Event Dispatch Thread.
 */
public class Main {
    public static void main(String[] args) {
        // Run UI initialization on the Event Dispatch Thread (standard Swing practice)
        SwingUtilities.invokeLater(() -> {
            try {
                // Use the modern system look and feel if available
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                // If setting system look fails, default Swing look is used gracefully
                System.out.println("Using default Swing look and feel: " + e.getMessage());
            }

            // Create and show the UI window
            DeepfakeDetectorUI appWindow = new DeepfakeDetectorUI();
            appWindow.setVisible(true);
        });
    }
}
