import os
import glob
import time
import pickle
import numpy as np
import cv2
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import hog

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
CADICA_DIR = r"C:\Users\athar\Downloads\CADICA\CADICA\selectedVideos"
CROP_SIZE = (64, 64)
MAX_POSITIVES = 800
MAX_NEGATIVES = 1600

# HOG Parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "cadica_model.pkl")

def compute_iou(boxA, boxB):
    # box: (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def extract_hog(crop):
    # Ensure crop is grayscale
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Resize to fixed size
    crop = cv2.resize(crop, CROP_SIZE)
    
    features = hog(
        crop,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        channel_axis=None
    )
    return features

def extract_samples(patient_dirs, max_pos, max_neg):
    X = []
    y = []
    
    pos_count = 0
    neg_count = 0
    
    for p_dir in patient_dirs:
        gt_files = glob.glob(os.path.join(p_dir, '**', 'groundtruth', '*.txt'), recursive=True)
        for gt_file in gt_files:
            if pos_count >= max_pos and neg_count >= max_neg:
                break
                
            img_file = gt_file.replace('groundtruth', 'input').replace('.txt', '.png')
            if not os.path.exists(img_file):
                continue
                
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            h, w = img.shape
            
            with open(gt_file, 'r') as f:
                lines = f.readlines()
            
            gt_boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        bx, by, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        # Filter invalid boxes
                        if bw > 0 and bh > 0 and bx >= 0 and by >= 0 and bx+bw <= w and by+bh <= h:
                            gt_boxes.append((bx, by, bw, bh))
                    except ValueError:
                        continue
            
            # Positive samples
            for box in gt_boxes:
                if pos_count < max_pos:
                    bx, by, bw, bh = box
                    crop = img[by:by+bh, bx:bx+bw]
                    if crop.size > 0:
                        feat = extract_hog(crop)
                        X.append(feat)
                        y.append(1)
                        pos_count += 1
            
            # Negative samples (random crop with low IoU)
            attempts = 0
            while neg_count < max_neg and attempts < 10:
                attempts += 1
                # Sample random size
                bw = np.random.randint(20, min(100, w//2))
                bh = np.random.randint(20, min(100, h//2))
                bx = np.random.randint(0, max(1, w - bw))
                by = np.random.randint(0, max(1, h - bh))
                
                neg_box = (bx, by, bw, bh)
                
                # Check overlap
                overlap = False
                for gtb in gt_boxes:
                    if compute_iou(neg_box, gtb) > 0.1:
                        overlap = True
                        break
                
                if not overlap:
                    crop = img[by:by+bh, bx:bx+bw]
                    if crop.size > 0:
                        feat = extract_hog(crop)
                        X.append(feat)
                        y.append(0)
                        neg_count += 1
                        break

    return np.array(X), np.array(y)

def main():
    print("==================================================")
    print("ML CLASSIFIER TRAINING SCRIPT")
    print("==================================================")
    
    t0 = time.time()
    
    # 1. Patient-level split to prevent data leakage
    all_patients = sorted(glob.glob(os.path.join(CADICA_DIR, 'p*')))
    if not all_patients:
        print(f"ERROR: No patient directories found in {CADICA_DIR}")
        return
        
    print(f"Found {len(all_patients)} patient directories.")
    
    # Simple split: first 70% train, last 30% test
    split_idx = int(0.7 * len(all_patients))
    train_patients = all_patients[:split_idx]
    test_patients = all_patients[split_idx:]
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # 2. Extract features
    print("\nExtracting training samples...")
    X_train, y_train = extract_samples(train_patients, MAX_POSITIVES, MAX_NEGATIVES)
    print(f"Training samples: {len(X_train)} ({sum(y_train)} positive, {len(y_train) - sum(y_train)} negative)")
    
    print("Extracting testing samples...")
    # Use smaller limit for test to speed up
    X_test, y_test = extract_samples(test_patients, MAX_POSITIVES // 3, MAX_NEGATIVES // 3)
    print(f"Testing samples: {len(X_test)} ({sum(y_test)} positive, {len(y_test) - sum(y_test)} negative)")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: Not enough samples extracted.")
        return
        
    # 3. Train Logistic Regression
    print("\nTraining Logistic Regression model...")
    t_train_start = time.time()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    t_train_end = time.time()
    print(f"Training took {t_train_end - t_train_start:.2f} seconds.")
    
    # 4. Evaluate on held-out patient data
    print("\nEvaluating on held-out test data (No Data Leakage):")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # 5. Save Model
    print(f"\nSaving model and configuration to {MODEL_OUTPUT}")
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    
    model_data = {
        'model': clf,
        'config': {
            'crop_size': CROP_SIZE,
            'hog_orientations': HOG_ORIENTATIONS,
            'hog_pixels_per_cell': HOG_PIXELS_PER_CELL,
            'hog_cells_per_block': HOG_CELLS_PER_BLOCK
        },
        'metadata': {
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'pos_samples': int(sum(y_train)),
            'neg_samples': int(len(y_train) - sum(y_train)),
            'f1_score': float(f1)
        }
    }
    
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nTotal elapsed time: {time.time() - t0:.2f} seconds.")
    print("==================================================")
    print("TRAINING COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
