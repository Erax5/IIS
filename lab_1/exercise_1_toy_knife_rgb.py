#!/bin/python


# Put your import statements up here
import cv2


# We will encapsulate all the code inside this main() function.
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
        # (4) "Mask" the wide image you just created:
        #      For each pixel, check if it's within the expected range (min_R <= R <= max_R, etc.),
        #      and set it to 0 if it's OUTSIDE the range.
        # (5) Display the masked image in the CV2 window "Color Picker".

        # Your code here

        # ---------------------------------------------------------------- #

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC pressed
            break

    # ---------------------------------------------------------------- #
    # (6) Save the masked image to "masked_rgb.jpg".
    # (7) Print the `min_values` and `max_values`.

    # Your code here

    # ---------------------------------------------------------------- #


if __name__ == "__main__":
    main()
