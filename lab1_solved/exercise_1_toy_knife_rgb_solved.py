#!/bin/python


# Put your import statements up here
import cv2
import numpy as np
import imageio.v3 as iio


# We will encapsulate all the code inside this main() function.
def main():
    # ---------------------------------------------------------------- #
    # (1) Load "toy_knife.jpg"

    # One option: use ImageIO (loads RGB) and use cv2 to convert to BGR.
    # original = iio.imread("toy_knife.jpg")
    # original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    # Another option: use the cv2 load function (loads as BGR, doesn't work with video).
    original = cv2.imread("toy_knife.jpg")

    # (2) Make a copy that has half the values of the original.
    dimmed = original / 2

    # ---------------------------------------------------------------- #

    cv2.namedWindow("Color Picker")

    min_values = [ 0, 0, 0 ]
    max_values = [ 255, 255, 255 ]

    def on_trackbar(_):
        min_values[0] = cv2.getTrackbarPos("R min", "Color Picker")
        max_values[0] = cv2.getTrackbarPos("R max", "Color Picker")
        min_values[1] = cv2.getTrackbarPos("G min", "Color Picker")
        max_values[1] = cv2.getTrackbarPos("G max", "Color Picker")
        min_values[2] = cv2.getTrackbarPos("B min", "Color Picker")
        max_values[2] = cv2.getTrackbarPos("B max", "Color Picker")

    cv2.createTrackbar("R min", "Color Picker",   0, 255, on_trackbar)
    cv2.createTrackbar("R max", "Color Picker", 255, 255, on_trackbar)
    cv2.createTrackbar("G min", "Color Picker",   0, 255, on_trackbar)
    cv2.createTrackbar("G max", "Color Picker", 255, 255, on_trackbar)
    cv2.createTrackbar("B min", "Color Picker",   0, 255, on_trackbar)
    cv2.createTrackbar("B max", "Color Picker", 255, 255, on_trackbar)

    while True:
        # ---------------------------------------------------------------- #
        # (3) Create an image that is twice as wide as the original.
        #     The left side must contain the pixels from the original image,
        #     and the right pixels must contain the pixels from the half-intensity image.
        (h, w, c) = original.shape
        masked = np.zeros_like(original, shape=(h, w*2, c))
        masked[:, :w, :] = original
        masked[:, w:, :] = dimmed

        # (4) "Mask" the wide image you just created:
        #      For each pixel, check if it's within the expected range (min_R <= R <= max_R, etc.),
        #      and set it to 0 if it's OUTSIDE the range.
        masked[masked[:,:,0] < min_values[0]] = 0
        masked[masked[:,:,1] < min_values[1]] = 0
        masked[masked[:,:,2] < min_values[2]] = 0

        masked[masked[:,:,0] > max_values[0]] = 0
        masked[masked[:,:,1] > max_values[1]] = 0
        masked[masked[:,:,2] > max_values[2]] = 0

        # (5) Display the masked image in the CV2 window "Color Picker".
        cv2.imshow("Color Picker", masked)

        # ---------------------------------------------------------------- #

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC pressed
            break

    # ---------------------------------------------------------------- #
    # (6) Save the masked image to "masked_rgb.jpg".

    # One option: convert back to RGB and use ImageIO.
    # masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    # iio.imwrite("masked_rgb.jpg", masked)

    # Another option: Use the cv2 function (expects BGR, doesn't work with video).
    cv2.imwrite("masked_rgb.jpg", masked)

    # (7) Print the `min_values` and `max_values`.
    print(f"Min Values (RGB): {min_values}")
    print(f"Max Values (RGB): {max_values}")

    # ---------------------------------------------------------------- #


if __name__ == "__main__":
    main()
