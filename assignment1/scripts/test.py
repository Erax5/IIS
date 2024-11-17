import cv2
import numpy as np
import imageio.v3 as iio
import os
from IPython.display import Image
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS

images_path = "../dataset/images/"
images = ['arguing.jpg', 'back-off.jpg']
output_path = '../processed/images/'


def main():
    detector = Detector(device="cuda")
    for image in os.listdir(images_path):
        curr_image_path = images_path + image
        
        frame = cv2.imread(curr_image_path)
        faces = detector.detect_faces(frame)
        landmarks = detector.detect_landmarks(frame, faces)
        emotions = detector.detect_emotions(frame, faces, landmarks)
        aus = detector.detect_aus(frame, faces, landmarks)
        
        faces = faces[0]
        landmarks = landmarks[0]
        emotions = emotions[0]
        aus = aus[0]

        strongest_emotion = emotions.argmax(axis=1)

        for (face, top_emo) in zip(faces, strongest_emotion):
            (x0, y0, x1, y1, p) = face
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
            cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

        cv2.imwrite(output_path + image, frame)
main()