import os
import sys
import glob
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.detector import detect_candidates
from pipeline.tracker import TemporalTracker, _compute_iou
from pipeline.ml_model import MLClassifier

def main():
    print("==================================================")
    print("DIAGNOSTIC COMPARISON SCRIPT")
    print("==================================================")
    
    clf = MLClassifier()
    if not clf.is_loaded:
        print("ERROR: ML model failed to load!")
        return

    # Find a valid frame with groundtruth
    cadica_dir = r"C:\Users\athar\Downloads\CADICA\CADICA\selectedVideos"
    gt_files = glob.glob(os.path.join(cadica_dir, '**', 'groundtruth', '*.txt'), recursive=True)
    
    selected_img = None
    selected_gt = None
    selected_box = None
    
    for gt_file in gt_files:
        img_file = gt_file.replace('groundtruth', 'input').replace('.txt', '.png')
        if not os.path.exists(img_file):
            continue
            
        with open(gt_file, 'r') as f:
            lines = f.readlines()
            
        valid_box = None
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    bx, by, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    if bw > 0 and bh > 0:
                        valid_box = (bx, by, bw, bh)
                        break
                except ValueError:
                    continue
                    
        if valid_box:
            selected_img = img_file
            selected_gt = gt_file
            selected_box = valid_box
            break
            
    if not selected_img:
        print("Could not find a valid test image with groundtruth.")
        return
        
    print(f"Selected Frame: {selected_img}")
    print(f"Groundtruth File: {selected_gt}")
    print(f"Groundtruth Box (x, y, w, h): {selected_box}")
    
    frame = cv2.imread(selected_img, cv2.IMREAD_GRAYSCALE)
    
    # 1. Evaluate Groundtruth Box directly
    bx, by, bw, bh = selected_box
    gt_crop = frame[by:by+bh, bx:bx+bw]
    gt_score = clf.predict_crop(gt_crop)
    
    print(f"Groundtruth AI Score: {gt_score:.3f} ({gt_score*100:.1f}%)")
    
    # 2. Evaluate OpenCV Candidates
    dets = detect_candidates(frame, frame_idx=0, ml_model=clf)
    
    best_iou = 0.0
    best_det = None
    
    for det in dets:
        iou = _compute_iou(selected_box, det.bbox)
        if iou > best_iou:
            best_iou = iou
            best_det = det
            
    if best_det:
        print(f"\nBest OpenCV Candidate Box: {best_det.bbox}")
        print(f"IoU with Groundtruth: {best_iou:.3f}")
        print(f"Candidate AI Score: {best_det.ml_score:.3f} ({best_det.ml_score*100:.1f}%)")
    else:
        print("\nNo overlapping OpenCV candidates found (IoU = 0.0).")

if __name__ == "__main__":
    main()
