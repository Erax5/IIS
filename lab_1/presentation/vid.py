#!/bin/env -S python3 -u


import numpy as np
import imageio.v3 as iio
from camera_widget import cv2_vid


def main():
    fps, vid = cv2_vid()
    iio.imwrite("vid.mp4", vid, fps=fps)


if __name__ == "__main__":
    main()
