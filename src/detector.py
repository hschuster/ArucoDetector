import cv2
import math
import time


class ArucoDetector:

    def __init__(self):

        # Dictionary wählen
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_1000
        )

        self.parameters = cv2.aruco.DetectorParameters()

        # Neuer Detector (OpenCV >= 4.7)
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            self.parameters
        )

        # Bewegungsverfolgung
        self.last_position = None
        self.last_time = None

    # ------------------------------------------------------------------------------------------------------------------
    def detect(self,
               frame,
               target_radius=50,
               servo_gain_x=0.1,
               servo_gain_y=0.1):

        """
        Erkennt Marker und zeichnet:
        - roten/grünen Rahmen
        - Pfeil zur Bildmitte
        - Distanzanzeige
        - Marker-ID
        - Geschwindigkeit
        - Servo-/Motorwerte
        - Zielkreis

        Es wird nur der Marker verfolgt,
        der dem Bildmittelpunkt am nächsten ist.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = self.detector.detectMarkers(gray)

        # Bildmittelpunkt
        frame_height, frame_width = frame.shape[:2]

        center_x = frame_width // 2
        center_y = frame_height // 2

        frame_center = (center_x, center_y)

        # Zielkreis zeichnen
        cv2.circle(
            frame,
            frame_center,
            target_radius,
            (255, 0, 0),
            2
        )

        # Mittelpunkt markieren
        cv2.circle(
            frame,
            frame_center,
            5,
            (255, 255, 255),
            -1
        )

        if ids is not None:

            nearest_marker = None
            nearest_distance = float("inf")

            # ----------------------------------------------------------------------------------------------------------
            # Nächsten Marker bestimmen
            # ----------------------------------------------------------------------------------------------------------

            for i, marker_corners in enumerate(corners):

                pts = marker_corners.reshape((4, 2)).astype(int)

                marker_center_x = int(pts[:, 0].mean())
                marker_center_y = int(pts[:, 1].mean())

                distance = math.sqrt(
                    (marker_center_x - center_x) ** 2 +
                    (marker_center_y - center_y) ** 2
                )

                if distance < nearest_distance:

                    nearest_distance = distance

                    nearest_marker = {
                        "index": i,
                        "pts": pts,
                        "center_x": marker_center_x,
                        "center_y": marker_center_y,
                        "distance": distance,
                        "id": int(ids[i][0])
                    }

            # ----------------------------------------------------------------------------------------------------------
            # Nur nächsten Marker darstellen
            # ----------------------------------------------------------------------------------------------------------

            if nearest_marker is not None:

                pts = nearest_marker["pts"]

                marker_center_x = nearest_marker["center_x"]
                marker_center_y = nearest_marker["center_y"]

                marker_center = (
                    marker_center_x,
                    marker_center_y
                )

                distance = nearest_marker["distance"]

                marker_id = nearest_marker["id"]

                # ------------------------------------------------------------------------------------------------------
                # Farbe abhängig vom Abstand
                # ------------------------------------------------------------------------------------------------------

                if distance <= target_radius:
                    color = (0, 255, 0)
                    status_text = "LOCKED"
                else:
                    color = (0, 0, 255)
                    status_text = "TRACKING"

                # ------------------------------------------------------------------------------------------------------
                # Markerrahmen
                # ------------------------------------------------------------------------------------------------------

                cv2.polylines(
                    frame,
                    [pts],
                    True,
                    color,
                    3
                )

                # ------------------------------------------------------------------------------------------------------
                # Mittelpunkt Marker
                # ------------------------------------------------------------------------------------------------------

                cv2.circle(
                    frame,
                    marker_center,
                    6,
                    color,
                    -1
                )

                # ------------------------------------------------------------------------------------------------------
                # Pfeil zur Bildmitte
                # ------------------------------------------------------------------------------------------------------

                cv2.arrowedLine(
                    frame,
                    frame_center,
                    marker_center,
                    color,
                    3,
                    tipLength=0.05
                )

                # ------------------------------------------------------------------------------------------------------
                # Distanz anzeigen
                # ------------------------------------------------------------------------------------------------------

                cv2.putText(
                    frame,
                    f"Dist: {distance:.1f}px",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                # ------------------------------------------------------------------------------------------------------
                # Marker-ID anzeigen
                # ------------------------------------------------------------------------------------------------------

                cv2.putText(
                    frame,
                    f"ID: {marker_id}",
                    (marker_center_x + 10, marker_center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                # ------------------------------------------------------------------------------------------------------
                # Geschwindigkeit berechnen
                # ------------------------------------------------------------------------------------------------------

                current_time = time.time()

                speed = 0.0

                if self.last_position is not None and self.last_time is not None:

                    delta_x = marker_center_x - self.last_position[0]
                    delta_y = marker_center_y - self.last_position[1]

                    delta_distance = math.sqrt(
                        delta_x ** 2 +
                        delta_y ** 2
                    )

                    delta_time = current_time - self.last_time

                    if delta_time > 0:
                        speed = delta_distance / delta_time

                self.last_position = (
                    marker_center_x,
                    marker_center_y
                )

                self.last_time = current_time

                cv2.putText(
                    frame,
                    f"Speed: {speed:.1f}px/s",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                # ------------------------------------------------------------------------------------------------------
                # Servo-/Motorsteuerung
                # ------------------------------------------------------------------------------------------------------

                error_x = marker_center_x - center_x
                error_y = marker_center_y - center_y

                servo_x = error_x * servo_gain_x
                servo_y = error_y * servo_gain_y

                cv2.putText(
                    frame,
                    f"Servo X: {servo_x:.2f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Servo Y: {servo_y:.2f}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # ------------------------------------------------------------------------------------------------------
                # Status
                # ------------------------------------------------------------------------------------------------------

                cv2.putText(
                    frame,
                    status_text,
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    3
                )

        return frame