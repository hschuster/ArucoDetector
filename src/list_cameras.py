import cv2
import argparse
from detector import ArucoDetector


# ------------------------------------------------------------
def main():

    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            print(f"Camera detected: {index}-{cap.getBackendName()}")
        cap.release()
        index += 1

if __name__ == "__main__":
    main()
