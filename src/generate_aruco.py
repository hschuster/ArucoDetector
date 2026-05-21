import cv2
import numpy as np
import argparse
import math


# ------------------------------------------------------------
def get_dictionary(dict_name: str):
    dict_map = {
        "4x4_50": cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "5x5_250": cv2.aruco.DICT_5X5_250,
        "6x6_250": cv2.aruco.DICT_6X6_250,
        "6x6_1000": cv2.aruco.DICT_6X6_1000,
        "7x7_250": cv2.aruco.DICT_7X7_250,
    }

    if dict_name not in dict_map:
        raise ValueError(f"Unbekanntes Dictionary: {dict_name}")

    return cv2.aruco.getPredefinedDictionary(dict_map[dict_name])


# ------------------------------------------------------------
def generate_marker(dictionary, marker_id, size, border):
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)

    # Weißer Rand (wichtig für Detection!)
    marker = cv2.copyMakeBorder(
        marker,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return marker


# ------------------------------------------------------------
def create_grid(
        dictionary,
        ids,
        marker_size=200,
        border=20,
        spacing=40,
        columns=4,
        show_ids=True
):
    rows = math.ceil(len(ids) / columns)

    single_size = marker_size + 2 * border

    width = columns * single_size + (columns + 1) * spacing
    height = rows * single_size + (rows + 1) * spacing

    canvas = np.ones((height, width), dtype=np.uint8) * 255

    for index, marker_id in enumerate(ids):
        row = index // columns
        col = index % columns

        marker_img = generate_marker(dictionary, marker_id, marker_size, border)

        y = spacing + row * (single_size + spacing)
        x = spacing + col * (single_size + spacing)

        canvas[y:y + single_size, x:x + single_size] = marker_img

        if show_ids:
            text = f"ID {marker_id}"
            cv2.putText(
                canvas,
                text,
                (x + 5, y + single_size - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,),
                1,
                cv2.LINE_AA
            )

    return canvas


# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dict", default="6x6_250", help="Dictionary (z.B. 6x6_250)")
    parser.add_argument("--count", type=int, default=12, help="Anzahl Marker")
    parser.add_argument("--start-id", type=int, default=0, help="Start-ID")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--marker-size", type=int, default=200)
    parser.add_argument("--border", type=int, default=20)
    parser.add_argument("--spacing", type=int, default=40)
    parser.add_argument("--output", default="images/aruco_grid.png")
    parser.add_argument("--no-labels", action="store_true")

    args = parser.parse_args()

    dictionary = get_dictionary(args.dict)

    ids = list(range(args.start_id, args.start_id + args.count))

    grid = create_grid(
        dictionary=dictionary,
        ids=ids,
        marker_size=args.marker_size,
        border=args.border,
        spacing=args.spacing,
        columns=args.columns,
        show_ids=not args.no_labels
    )

    cv2.imwrite(args.output, grid)

    print(f"Bild gespeichert: {args.output}")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()