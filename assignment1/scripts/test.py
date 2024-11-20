import cv2
import numpy as np
import imageio.v3 as iio
import os
from IPython.display import Image
from feat import Detector, Fex
from feat.utils import FEAT_EMOTION_COLUMNS
import pandas as pd 

images_path = "../dataset/images/"
images = ['arguing.jpg', 'back-off.jpg']
output_path = '../processed/images/'


def main():
    fex = Fex()
    detector = Detector(device="cuda")
    au_data = []

    for image in os.listdir(images_path):
        curr_image_path = images_path + image
        
        frame = cv2.imread(curr_image_path)
        faces = detector.detect_faces(frame) 
        #list of lists with the same length as the number of frames. 
        #Each list item is a list containing the (x1, y1, x2, y2) coordinates of each detected face in that frame.

        landmarks = detector.detect_landmarks(frame, faces)
        emotions = detector.detect_emotions(frame, faces, landmarks)
        aus = detector.detect_aus(frame, landmarks)

        faces = faces[0]
        landmarks = landmarks[0]
        emotions = emotions[0]
        aus = aus[0]

        strongest_emotion = emotions.argmax(axis=1)
        print("image: ", image)
        for i, (face, top_emo) in enumerate(zip(faces, strongest_emotion)):
            print("face: ", face)
            # print("face: ", face)
            (x0, y0, x1, y1, p) = face
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
            cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

            au_row = {"file": image, "face": i }
            for j, au_value in enumerate(aus[i]):
                au_row[f"AU{j + 1}"] = au_value
            au_data.append(au_row)
            # Save AU activations to CSV
        au_df = pd.DataFrame(au_data)
        au_df.to_csv('../processed/aus.csv', index=False)
        print(f"AU activations saved to ../processed/aus.csv")
        print(aus)
        cv2.imwrite(output_path + image, frame)
main()