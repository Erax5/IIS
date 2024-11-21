import cv2
import os
from feat import Detector, Fex
from feat.utils import FEAT_EMOTION_COLUMNS
import pandas as pd 
import matplotlib.pyplot as plt
import time
# används ej:
# from IPython.display import Image
# import numpy as np
# import imageio.v3 as iio

images_path = "dataset/images/"
output_path = 'processed/images/'
aus_path = 'processed/aus.csv'
annotations_path = 'dataset/annotations.csv'
annotations_df = pd.read_csv(annotations_path)

def main():
    fex = Fex()
    detector = Detector(device="cuda")
    au_data = []
    positive_data = []
    negative_data = []
    
    start_time = time.time()
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
        for i, (face, top_emo) in enumerate(zip(faces, strongest_emotion)):
            (x0, y0, x1, y1, p) = face
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
            cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

            au_row = {"file": image, "face": i }
            for j, au_value in enumerate(aus[i]):
                au_row[f"AU{j + 1}"] = au_value
            au_data.append(au_row)
            
        # Save image with added bounding box and emotion label
        cv2.imwrite(output_path + image, frame)

    # Save AU activations to CSV outside the loop
    au_df = pd.DataFrame(au_data)
    au_df.to_csv(aus_path, index=False)

    # ----------------- Split data into positive and negative -----------------

    au_df['file'] = au_df['file'].apply(lambda x: os.path.splitext(x)[0])
    positive_data = []
    negative_data = []
    
    for i, au_row in au_df.iterrows():
        if au_row.isnull().values.any():
            continue
        file_name = au_row["file"]
        
        valence = annotations_df[annotations_df["file"] == file_name]["valence"].values[0]
        if valence == "positive":
            positive_data.append(au_row)
        else:
            negative_data.append(au_row)

    positive_df = pd.DataFrame(positive_data)
    negative_df = pd.DataFrame(negative_data)

    positive_df.to_csv('processed/positive_data.csv', index=False)
    negative_df.to_csv('processed/negative_data.csv', index=False)

    # -------------------------------------------------------------------------
    
    # ----------------- Calculate mean of each AU for positive and negative data and plot -----------------

    # go through negative_data.csv and positive_data.csv and calculate the mean of each AU
    # for each au, calculate the absolute difference between the mean of the positive and negative data
    # sort the AUs by the absolute difference
    # plot and save the sorted list of AUs

    abs_diff = {}

    for col in negative_df.columns:
        if col == "file" or col == "face":
            continue

        positive_mean = positive_df[col].mean()
        negative_mean = negative_df[col].mean()
        absolute_difference = abs(positive_mean - negative_mean)
        
        abs_diff[col] = absolute_difference

    #sort the AUs by the absolute difference
    abs_diff = dict(sorted(abs_diff.items(), key=lambda item: item[1], reverse=True))

    # Create a DataFrame with AU names and their absolute mean values
    abs_diff_df = pd.DataFrame([abs_diff])
    abs_diff_df.to_csv('processed/abs_diff.csv', index=False)

    # Plot all values
    plt.plot(list(abs_diff.keys()), list(abs_diff.values()), 'b.')

    # Highlight the 6 highest values
    top_6_aus = list(abs_diff.keys())[:6]
    top_6_values = list(abs_diff.values())[:6]
    plt.plot(top_6_aus, top_6_values, 'r.') 
    plt.xticks(rotation=45)
    plt.xlabel('AU')
    plt.ylabel('Absolute difference')
    plt.title('Absolute difference between positive and negative AU')
    # plt.show()
    plt.savefig('processed/au_visualization.png')

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
main()