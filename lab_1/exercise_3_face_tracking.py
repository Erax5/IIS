#!/bin/python


import cv2


def main():
    face_tracker = cv2.CascadeClassifier("frontal_face_features.xml")

    # ---------------------------------------------------------------- #
    # Use an OpenCV window to display a live feed of your webcam.
    # When you press SPACE, the program should toggle between "recording" mode and "stopped" mode.
    # * In "stopped" mode, you should only display the live feed, with no modifications.
    # * In "recording" mode, you should display a red circle on the top left corner to show that the mode is active.
    #   You should use the `face_tracker` to find any faces in the current frame, and display
    #   a green rectangle where the detected faces are (draw the outline, not a solid rectangle).

    # Your code here

    # ---------------------------------------------------------------- #

if __name__ == "__main__":
    main()
