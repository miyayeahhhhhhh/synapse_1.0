"""
pipeline/ml_model.py
--------------------
Real machine learning inference component for Synapse 1.0.

Loads a pre-trained scikit-learn Logistic Regression model (trained on CADICA)
and classifies candidate crops extracted by the OpenCV heuristic detector.
"""

import os
import cv2
import pickle
from skimage.feature import hog

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cadica_model.pkl")

class MLClassifier:
    def __init__(self):
        self.model = None
        self.config = None
        self.is_loaded = False
        
        self.load_model()
        
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: ML model not found at {MODEL_PATH}")
            return False
            
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.config = data['config']
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading ML model: {e}")
            return False
            
    def predict_crop(self, crop):
        """
        Extracts HOG features from a grayscale crop and predicts the probability
        of it being a CADICA lesion-like candidate.
        
        Returns probability of positive class (float between 0 and 1).
        """
        if not self.is_loaded or self.model is None or self.config is None:
            return 0.0
            
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
        # Resize to fixed size used during training
        crop = cv2.resize(crop, tuple(self.config['crop_size']))
        
        # Extract features
        features = hog(
            crop,
            orientations=self.config['hog_orientations'],
            pixels_per_cell=tuple(self.config['hog_pixels_per_cell']),
            cells_per_block=tuple(self.config['hog_cells_per_block']),
            block_norm='L2-Hys',
            visualize=False,
            channel_axis=None
        )
        
        # predict_proba returns [[prob_0, prob_1]]
        # We want the probability of class 1 (positive)
        prob = self.model.predict_proba([features])[0][1]
        return float(prob)
