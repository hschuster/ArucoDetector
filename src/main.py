import cv2
import argparse

from detector import ArucoDetector


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

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detector.detect(
            frame,
            target_radius=target_radius
        )

        cv2.imshow(
            "Aruco Detection",
            result
        )

        key = cv2.waitKey(1) & 0xFF

        # ESC zum Beenden
        if key == 27:
            break

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
