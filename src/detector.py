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
        - alle Marker mit Rahmen
        - Pfeil zur Bildmitte
        - Marker-ID
        - Distanz

        Zusätzliche Infos (Speed, Servo, LOCKED)
        werden nur für den nächsten Marker angezeigt.
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
            # Alle Marker zeichnen + nächsten Marker bestimmen
            # ----------------------------------------------------------------------------------------------------------

            for i, marker_corners in enumerate(corners):

                pts = marker_corners.reshape((4, 2)).astype(int)

                marker_center_x = int(pts[:, 0].mean())
                marker_center_y = int(pts[:, 1].mean())

                marker_center = (
                    marker_center_x,
                    marker_center_y
                )

                distance = math.sqrt(
                    (marker_center_x - center_x) ** 2 +
                    (marker_center_y - center_y) ** 2
                )

                marker_id = int(ids[i][0])

                # ------------------------------------------------------------------------------------------------------
                # Nächsten Marker merken
                # ------------------------------------------------------------------------------------------------------

                if distance < nearest_distance:

                    nearest_distance = distance

                    nearest_marker = {
                        "center_x": marker_center_x,
                        "center_y": marker_center_y,
                        "distance": distance,
                        "id": marker_id
                    }

                # ------------------------------------------------------------------------------------------------------
                # Farbe abhängig vom Abstand
                # ------------------------------------------------------------------------------------------------------

                if distance <= target_radius:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

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
                # Mittelpunkt
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
                    2,
                    tipLength=0.05
                )

                # ------------------------------------------------------------------------------------------------------
                # Marker-ID
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
                # Distanz
                # ------------------------------------------------------------------------------------------------------

                cv2.putText(
                    frame,
                    f"{distance:.0f}px",
                    (marker_center_x + 10, marker_center_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # ----------------------------------------------------------------------------------------------------------
            # Zusätzliche Infos nur für nächsten Marker
            # ----------------------------------------------------------------------------------------------------------

            if nearest_marker is not None:

                marker_center_x = nearest_marker["center_x"]
                marker_center_y = nearest_marker["center_y"]

                distance = nearest_marker["distance"]

                # ------------------------------------------------------------------------------------------------------
                # Status
                # ------------------------------------------------------------------------------------------------------

                if distance <= target_radius:
                    color = (0, 255, 0)
                    status_text = "LOCKED"
                else:
                    color = (0, 0, 255)
                    status_text = "TRACKING"

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

                # ------------------------------------------------------------------------------------------------------
                # Servo-/Motorsteuerung
                # ------------------------------------------------------------------------------------------------------

                error_x = marker_center_x - center_x
                error_y = marker_center_y - center_y

                servo_x = error_x * servo_gain_x
                servo_y = error_y * servo_gain_y

                # ------------------------------------------------------------------------------------------------------
                # Overlay-Infos
                # ------------------------------------------------------------------------------------------------------

                cv2.putText(
                    frame,
                    f"Nearest ID: {nearest_marker['id']}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Distance: {distance:.1f}px",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Speed: {speed:.1f}px/s",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Servo X: {servo_x:.2f}",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Servo Y: {servo_y:.2f}",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    status_text,
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    3
                )

        return frame