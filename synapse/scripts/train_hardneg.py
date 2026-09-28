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
MAX_HARD_NEGATIVES = 1000

# HOG Parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

OLD_MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "cadica_model.pkl")
NEW_MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "cadica_model_hardneg.pkl")

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def extract_hog(crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, CROP_SIZE)
    features = hog(
        crop, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys', visualize=False, channel_axis=None
    )
    return features

def get_gt_boxes(gt_file, w, h):
    gt_boxes = []
    with open(gt_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    bx, by, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    if bw > 0 and bh > 0 and bx >= 0 and by >= 0 and bx+bw <= w and by+bh <= h:
                        gt_boxes.append((bx, by, bw, bh))
                except ValueError:
                    continue
    return gt_boxes

def extract_samples(patient_dirs, max_pos, max_neg):
    X, y = [], []
    pos_count, neg_count = 0, 0
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
            gt_boxes = get_gt_boxes(gt_file, w, h)
            
            for box in gt_boxes:
                if pos_count < max_pos:
                    bx, by, bw, bh = box
                    crop = img[by:by+bh, bx:bx+bw]
                    if crop.size > 0:
                        X.append(extract_hog(crop))
                        y.append(1)
                        pos_count += 1
                        
            attempts = 0
            while neg_count < max_neg and attempts < 10:
                attempts += 1
                bw, bh = np.random.randint(20, min(100, w//2)), np.random.randint(20, min(100, h//2))
                bx, by = np.random.randint(0, max(1, w - bw)), np.random.randint(0, max(1, h - bh))
                neg_box = (bx, by, bw, bh)
                
                overlap = any(compute_iou(neg_box, gtb) > 0.1 for gtb in gt_boxes)
                if not overlap:
                    crop = img[by:by+bh, bx:bx+bw]
                    if crop.size > 0:
                        X.append(extract_hog(crop))
                        y.append(0)
                        neg_count += 1
                        break
    return np.array(X), np.array(y)

def mine_hard_negatives(patient_dirs, old_model, max_hard_neg):
    X_hn = []
    y_hn = []
    
    count = 0
    for p_dir in patient_dirs:
        if count >= max_hard_neg:
            break
        gt_files = glob.glob(os.path.join(p_dir, '**', 'groundtruth', '*.txt'), recursive=True)
        # shuffle gt_files to get variety? Or just process them.
        for gt_file in gt_files:
            if count >= max_hard_neg:
                break
            img_file = gt_file.replace('groundtruth', 'input').replace('.txt', '.png')
            if not os.path.exists(img_file):
                continue
            
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            h, w = img.shape
            gt_boxes = get_gt_boxes(gt_file, w, h)
            
            # Sliding window to find hard negatives
            ww, wh = 64, 64
            stride = 48
            for y in range(0, h - wh + 1, stride):
                for x in range(0, w - ww + 1, stride):
                    if count >= max_hard_neg:
                        break
                    window = (x, y, ww, wh)
                    overlap = any(compute_iou(window, gtb) > 0.1 for gtb in gt_boxes)
                    if not overlap:
                        crop = img[y:y+wh, x:x+ww]
                        feat = extract_hog(crop)
                        prob = old_model.predict_proba([feat])[0][1]
                        if prob >= 0.80:
                            X_hn.append(feat)
                            y_hn.append(0)
                            count += 1
                            
    return np.array(X_hn), np.array(y_hn)

def sliding_window_diagnostic(img_path, gt_box, clf):
    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None, None
        
    gt_crop = frame[gt_box[1]:gt_box[1]+gt_box[3], gt_box[0]:gt_box[0]+gt_box[2]]
    gt_feat = extract_hog(gt_crop)
    gt_score = clf.predict_proba([gt_feat])[0][1]
    
    h, w = frame.shape
    ww, wh = 64, 64
    stride = 32
    max_bg_score = 0.0
    for y in range(0, h - wh + 1, stride):
        for x in range(0, w - ww + 1, stride):
            window = (x, y, ww, wh)
            if compute_iou(window, gt_box) < 0.1:
                crop = frame[y:y+wh, x:x+ww]
                feat = extract_hog(crop)
                score = clf.predict_proba([feat])[0][1]
                if score > max_bg_score:
                    max_bg_score = score
                    
    return gt_score, max_bg_score

def main():
    print("==================================================")
    print("HARD NEGATIVE MINING & RETRAINING")
    print("==================================================")
    
    t0 = time.time()
    
    all_patients = sorted(glob.glob(os.path.join(CADICA_DIR, 'p*')))
    split_idx = int(0.7 * len(all_patients))
    train_patients = all_patients[:split_idx]
    test_patients = all_patients[split_idx:]
    
    print("\n1. Loading old model...")
    with open(OLD_MODEL_OUTPUT, 'rb') as f:
        old_data = pickle.load(f)
    old_model = old_data['model']
    
    print("\n2. Extracting original train/test split...")
    X_train_orig, y_train_orig = extract_samples(train_patients, MAX_POSITIVES, MAX_NEGATIVES)
    X_test, y_test = extract_samples(test_patients, MAX_POSITIVES // 3, MAX_NEGATIVES // 3)
    
    print(f"Original Training Samples: {len(X_train_orig)}")
    print(f"Test Samples: {len(X_test)}")
    
    print("\n3. Mining hard negatives from training patients...")
    t_mine = time.time()
    X_hn, y_hn = mine_hard_negatives(train_patients, old_model, MAX_HARD_NEGATIVES)
    print(f"Mined {len(X_hn)} hard negatives in {time.time() - t_mine:.2f}s.")
    
    if len(X_hn) == 0:
        print("No hard negatives found! The old model might already be perfect?")
        return
        
    X_train_new = np.vstack((X_train_orig, X_hn))
    y_train_new = np.concatenate((y_train_orig, y_hn))
    
    print(f"New Training Samples: {len(X_train_new)} ({sum(y_train_new)} pos, {len(y_train_new)-sum(y_train_new)} neg)")
    
    print("\n4. Retraining Logistic Regression model...")
    t_train = time.time()
    new_clf = LogisticRegression(max_iter=1000, random_state=42)
    new_clf.fit(X_train_new, y_train_new)
    print(f"Training took {time.time() - t_train:.2f}s.")
    
    print("\n5. Evaluating Old Model on Test Set...")
    y_pred_old = old_model.predict(X_test)
    acc_old = accuracy_score(y_test, y_pred_old)
    prec_old = precision_score(y_test, y_pred_old, zero_division=0)
    rec_old = recall_score(y_test, y_pred_old, zero_division=0)
    f1_old = f1_score(y_test, y_pred_old, zero_division=0)
    print(f"  Accuracy:  {acc_old:.4f}")
    print(f"  Precision: {prec_old:.4f}")
    print(f"  Recall:    {rec_old:.4f}")
    print(f"  F1 Score:  {f1_old:.4f}")
    
    print("\n6. Evaluating New Model on Test Set...")
    y_pred_new = new_clf.predict(X_test)
    acc_new = accuracy_score(y_test, y_pred_new)
    prec_new = precision_score(y_test, y_pred_new, zero_division=0)
    rec_new = recall_score(y_test, y_pred_new, zero_division=0)
    f1_new = f1_score(y_test, y_pred_new, zero_division=0)
    print(f"  Accuracy:  {acc_new:.4f}")
    print(f"  Precision: {prec_new:.4f}")
    print(f"  Recall:    {rec_new:.4f}")
    print(f"  F1 Score:  {f1_new:.4f}")
    
    print("\n7. Diagnostic Frames Re-evaluation...")
    frames = [
        ("p1_v10_00014.png", r"C:\Users\athar\Downloads\CADICA\CADICA\selectedVideos\p1\v10\input\p1_v10_00014.png", (259, 128, 34, 33)),
        ("p1_v10_00018.png", r"C:\Users\athar\Downloads\CADICA\CADICA\selectedVideos\p1\v10\input\p1_v10_00018.png", (263, 132, 54, 75))
    ]
    
    for name, path, gt_box in frames:
        print(f"\n--- {name} ---")
        gt_old, bg_old = sliding_window_diagnostic(path, gt_box, old_model)
        gt_new, bg_new = sliding_window_diagnostic(path, gt_box, new_clf)
        
        print("OLD MODEL:")
        print(f"  Exact GT Score: {gt_old:.3f}")
        print(f"  Highest BG Score: {bg_old:.3f}")
        print("NEW MODEL:")
        print(f"  Exact GT Score: {gt_new:.3f}")
        print(f"  Highest BG Score: {bg_new:.3f}")
        
    print("\nSaving new model...")
    model_data = {
        'model': new_clf,
        'config': old_data['config'],
        'metadata': {
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'pos_samples': int(sum(y_train_new)),
            'neg_samples': int(len(y_train_new) - sum(y_train_new)),
            'f1_score': float(f1_new)
        }
    }
    with open(NEW_MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nTotal elapsed time: {time.time() - t0:.2f}s")
    print("==================================================")
    print("HARD NEGATIVE MINING COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
