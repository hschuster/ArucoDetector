import cv2
import argparse
from datetime import datetime

from detector import ArucoDetector


# ----------------------------------------------------------------------------------------------------------------------
def create_timestamp_filename():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"aruco_recording_{timestamp}.mp4"


# ----------------------------------------------------------------------------------------------------------------------
def create_screenshot_filename():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"aruco_screenshot_{timestamp}.png"


# ----------------------------------------------------------------------------------------------------------------------
def process_image(path,
                  target_radius):

    detector = ArucoDetector()

    frame = cv2.imread(path)

    if frame is None:
        print("Bild konnte nicht geladen werden")
        return

    result = detector.detect(
        frame,
        target_radius=target_radius
    )

    cv2.imshow("Aruco Detection", result)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------------------------------------------------
def process_video(source,
                  target_radius):

    detector = ArucoDetector()

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Videoquelle konnte nicht geöffnet werden")
        return

    writer = None
    recording = False

    print("Steuerung:")
    print("  ESC = Beenden")
    print("  R    = Aufnahme Start/Stop")
    print("  S    = Screenshot")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detector.detect(
            frame,
            target_radius=target_radius
        )

        # --------------------------------------------------------------------------------------------------------------
        # Aufnahme speichern
        # --------------------------------------------------------------------------------------------------------------

        if recording and writer is not None:

            writer.write(result)

            cv2.putText(
                result,
                "REC",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        # --------------------------------------------------------------------------------------------------------------
        # Anzeige
        # --------------------------------------------------------------------------------------------------------------

        cv2.imshow(
            "Aruco Detection",
            result
        )

        key = cv2.waitKey(1) & 0xFF

        # --------------------------------------------------------------------------------------------------------------
        # ESC = Ende
        # --------------------------------------------------------------------------------------------------------------

        if key == 27:
            break

        # --------------------------------------------------------------------------------------------------------------
        # R = Aufnahme toggeln
        # --------------------------------------------------------------------------------------------------------------

        elif key == ord('r'):

            if not recording:

                filename = create_timestamp_filename()

                fps = 20.0

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    filename,
                    fourcc,
                    fps,
                    (width, height)
                )

                recording = True

                print(f"Aufnahme gestartet: {filename}")

            else:

                recording = False

                if writer is not None:
                    writer.release()
                    writer = None

                print("Aufnahme beendet")

        # --------------------------------------------------------------------------------------------------------------
        # S = Screenshot
        # --------------------------------------------------------------------------------------------------------------

        elif key == ord('s'):

            screenshot_file = create_screenshot_filename()

            cv2.imwrite(
                screenshot_file,
                result
            )

            print(f"Screenshot gespeichert: {screenshot_file}")

    # ------------------------------------------------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------------------------------------------------

    if writer is not None:
        writer.release()

    cap.release()

    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        help="Pfad zu Bild"
    )

    parser.add_argument(
        "--video",
        help="Pfad zu Video"
    )

    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Webcam verwenden"
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=50,
        help="Zielradius rund um Bildmitte"
    )

    args = parser.parse_args()

    if args.image:

        process_image(
            args.image,
            args.radius
        )

    elif args.video:

        process_video(
            args.video,
            args.radius
        )

    elif args.webcam:

        process_video(
            1,  # 0=Laptop, 1=USB-Webcam
            args.radius
        )

    else:

        print(
            "Bitte Parameter angeben: "
            "--image | --video | --webcam"
        )


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()