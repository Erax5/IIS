#!/bin/python


# Put your import statements up here
import cv2


def main():
    # ---------------------------------------------------------------- #
    # (1) Load "toy_knife.jpg"
    # (2) Make a copy that has half the values of the original.

    # Your code here

    # ---------------------------------------------------------------- #

    cv2.namedWindow("Color Picker")

    min_values = [ 0, 0, 0 ]
    max_values = [ 255, 255, 255 ]

    def on_trackbar(_):
        min_values[0] = cv2.getTrackbarPos("H min", "Color Picker")
        max_values[0] = cv2.getTrackbarPos("H max", "Color Picker")
        min_values[1] = cv2.getTrackbarPos("S min", "Color Picker")
        max_values[1] = cv2.getTrackbarPos("S max", "Color Picker")
        min_values[2] = cv2.getTrackbarPos("V min", "Color Picker")
        max_values[2] = cv2.getTrackbarPos("V max", "Color Picker")

    cv2.createTrackbar("H min", "Color Picker",   0, 255, on_trackbar)
    cv2.createTrackbar("H max", "Color Picker", 255, 255, on_trackbar)
    cv2.createTrackbar("S min", "Color Picker",   0, 255, on_trackbar)
    cv2.createTrackbar("S max", "Color Picker", 255, 255, on_trackbar)
    cv2.createTrackbar("V min", "Color Picker",   0, 255, on_trackbar)
    cv2.createTrackbar("V max", "Color Picker", 255, 255, on_trackbar)

    while True:
        # ---------------------------------------------------------------- #
        # (3) Create an image that is twice as wide as the original.
        #     The left side must contain the pixels from the original image,
        #     and the right pixels must contain the pixels from the half-intensity image.
        # (4) Convert the wide image to HSV using OpenCV.
        # (5) Mask the wide image like you did in the RGB exercise,
        # BUT this time use the HSV values for masking.
        # (6) Display the masked image in the CV2 window "Color Picker".

        # Your code here

        # ---------------------------------------------------------------- #

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC pressed
            break

    # ---------------------------------------------------------------- #
    # (7) Save the masked image to "masked_hsv.jpg".
    # (8) Print the `min_values` and `max_values`.

    # Your code here

    # ---------------------------------------------------------------- #


if __name__ == "__main__":
    main()
