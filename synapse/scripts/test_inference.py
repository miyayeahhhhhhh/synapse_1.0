import os
import cv2
import pickle
import numpy as np
from pathlib import Path
from skimage.feature import hog

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pipeline", "cadica_model.pkl")

def extract_hog(crop, config):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, tuple(config['crop_size']))
    
    features = hog(
        crop,
        orientations=config['hog_orientations'],
        pixels_per_cell=tuple(config['hog_pixels_per_cell']),
        cells_per_block=tuple(config['hog_cells_per_block']),
        block_norm='L2-Hys',
        visualize=False,
        channel_axis=None
    )
    return features

def test_inference():
    print("==================================================")
    print("STANDALONE INFERENCE TEST")
    print("==================================================")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
        
    print(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        
    clf = model_data['model']
    config = model_data['config']
    
    print("Generating synthetic positive-like crop...")
    # Create a dummy crop that roughly simulates a dark region in a bright vessel
    crop = np.ones((100, 100), dtype=np.uint8) * 200
    crop[30:70, 30:70] = 50  # dark center
    
    feat = extract_hog(crop, config)
    prob = clf.predict_proba([feat])[0]
    
    print(f"Model classes: {clf.classes_}")
    print(f"Prediction probability: Class 0 = {prob[0]:.4f}, Class 1 = {prob[1]:.4f}")
    
    print("\nInference test successful.")
    
if __name__ == "__main__":
    test_inference()
