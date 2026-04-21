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
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Marker IDs zusätzlich als Text anzeigen
            for i, corner in enumerate(corners):
                c = corner[0]

                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())

                cv2.putText(
                    frame,
                    f"ID: {ids[i][0]}",
                    (center_x - 20, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        return frame