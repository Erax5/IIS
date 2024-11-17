#!/bin/env -S python3 -u


import imageio.v3 as iio
from camera_widget import cv2_snap


def main():
    snap = cv2_snap()
    iio.imwrite("snap.jpg", snap)


if __name__ == "__main__":
    main()
