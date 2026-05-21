import cv2
import numpy as np


class ArucoDetector:

    def __init__(self):
        # Dictionary wählen (Standard: 6x6 Marker)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        self.parameters = cv2.aruco.DetectorParameters()

        # Neuer Detector (OpenCV >= 4.7)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    # ------------------------------------------------------------
    def detect(self, frame):
        """
        Erkennt Marker und gibt annotiertes Bild zurück
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is not None:
            for marker_corners in corners:
                pts = marker_corners.reshape((4, 2)).astype(int)

                # Linien zwischen den 4 Punkten zeichnen
                for i in range(4):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[(i + 1) % 4])

                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        return frame