import cv2
import time
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.detector import detect_candidates
from pipeline.tracker import TemporalTracker
from pipeline.ml_model import MLClassifier

def main():
    print("==================================================")
    print("REAL CADICA ML INTEGRATION TEST")
    print("==================================================")

    clf = MLClassifier()
    if not clf.is_loaded:
        print("ERROR: ML model failed to load!")
        return

    # Grab a real CADICA image to test
    test_img_path = r"C:\Users\athar\Downloads\CADICA\CADICA\selectedVideos\p4\v2\input\p0_20.png"
    frame = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    
    if frame is None:
        print(f"Could not read test image at {test_img_path}")
        return

    print("Running detector with ML classifier on a real CADICA frame...")
    dets = detect_candidates(frame, frame_idx=0, ml_model=clf)

    print(f"Candidates found: {len(dets)}")
    for i, d in enumerate(dets):
        print(f"  Candidate {i+1}:")
        print(f"    bbox: {d.bbox}")
        print(f"    raw_score: {d.raw_score:.3f}")
        print(f"    ml_score: {d.ml_score:.3f}")

    tracker = TemporalTracker()
    tracks = tracker.update(0, dets)

    print("\nTracker output:")
    for t in tracks:
        print(f"  Lesion #{t.lesion_id} mean_ml_score: {t.mean_ml_score:.3f}")

    print("\nIntegration test successful!")

if __name__ == "__main__":
    main()
