import math
import cv2

class ArucoDetector:

    def __init__(self):
        # Dictionary wählen (Standard: 6x6 Marker)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

        self.parameters = cv2.aruco.DetectorParameters()

        # Neuer Detector (OpenCV >= 4.7)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    # ------------------------------------------------------------


    def detect(self, frame, target_radius=50):
        """
        Erkennt Marker und zeichnet:
        - roten/grünen Rahmen
        - Linie zur Bildmitte

        Grün = Marker befindet sich innerhalb des Zielradius
        Rot   = Marker außerhalb des Zielradius
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = self.detector.detectMarkers(gray)

        # Bildmittelpunkt
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2

        # Zielkreis visualisieren
        cv2.circle(
            frame,
            (center_x, center_y),
            target_radius,
            (255, 0, 0),
            2
        )

        if ids is not None:

            for marker_corners in corners:
                pts = marker_corners.reshape((4, 2)).astype(int)

                # Mittelpunkt des Markers berechnen
                marker_center_x = int(pts[:, 0].mean())
                marker_center_y = int(pts[:, 1].mean())

                # Abstand zur Bildmitte
                distance = math.sqrt(
                    (marker_center_x - center_x) ** 2 +
                    (marker_center_y - center_y) ** 2
                )

                # Farbe abhängig vom Abstand
                color = (0, 255, 0) if distance <= target_radius else (0, 0, 255)

                # Markerrahmen zeichnen
                cv2.polylines(
                    frame,
                    [pts],
                    True,
                    color,
                    2
                )

                # Linie zwischen Marker und Bildmitte
                cv2.line(
                    frame,
                    (marker_center_x, marker_center_y),
                    (center_x, center_y),
                    color,
                    2
                )

                # Mittelpunkt des Markers markieren
                cv2.circle(
                    frame,
                    (marker_center_x, marker_center_y),
                    5,
                    color,
                    -1
                )

        return frame
