import cv2
import time
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.ml_model import MLClassifier
from pipeline.ai_proposals import sliding_window_proposals, compute_iou

def test_frame(img_path, gt_box, clf, threshold=0.7):
    print(f"\n--- Testing Frame: {Path(img_path).name} ---")
    print(f"Groundtruth Box: {gt_box}")
    print(f"AI Threshold: {threshold}")
    
    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print("Failed to load image.")
        return
        
    t0 = time.time()
    proposals, windows_evaluated = sliding_window_proposals(
        frame, clf, window_size=(64, 64), stride=32, threshold=threshold
    )
    t1 = time.time()
    
    print(f"Windows evaluated: {windows_evaluated}")
    print(f"Processing time: {t1 - t0:.3f}s")
    
    if not proposals:
        print("No proposals found above threshold.")
        return
        
    # Sort proposals by score descending
    proposals.sort(key=lambda x: x[1], reverse=True)
    best_box, best_score = proposals[0]
    
    iou = compute_iou(gt_box, best_box)
    
    print(f"Highest AI score: {best_score:.3f}")
    print(f"Coordinates of highest-scoring proposal: {best_box}")
    print(f"IoU against groundtruth: {iou:.3f}")
    print(f"IoU >= 0.30: {'YES' if iou >= 0.30 else 'NO'}")

def main():
    print("==================================================")
    print("AI SLIDING WINDOW DIAGNOSTIC")
    print("==================================================")
    
    clf = MLClassifier()
    if not clf.is_loaded:
        print("Model failed to load.")
        return
        
    f1 = r"C:\Users\athar\Downloads\CADICA\CADICA\selectedVideos\p1\v10\input\p1_v10_00014.png"
    gt1 = (259, 128, 34, 33)
    
    f2 = r"C:\Users\athar\Downloads\CADICA\CADICA\selectedVideos\p1\v10\input\p1_v10_00018.png"
    gt2 = (263, 132, 54, 75)
    
    for thresh in [0.70, 0.80, 0.90]:
        print(f"\n================ Threshold {thresh} ================")
        test_frame(f1, gt1, clf, threshold=thresh)
        test_frame(f2, gt2, clf, threshold=thresh)

if __name__ == "__main__":
    main()
