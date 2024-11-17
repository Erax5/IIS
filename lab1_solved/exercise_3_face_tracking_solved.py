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

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Tracking")
    recording = False

    while True:
        ret, frame = cam.read()
        if not ret:
            print("OpenCV found an error reading the next frame.")
            break

        display = frame.copy()

        if recording:
            faces = face_tracker.detectMultiScale(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0))

            cv2.circle(display, (32, 32), 16, (0, 0, 255), -1)

        cv2.imshow("Face Tracking", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC pressed
            break
        elif key == 32: # SPACE pressed
            recording = not recording

    # ---------------------------------------------------------------- #


if __name__ == "__main__":
    main()
