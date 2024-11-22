import cv2
import os
from feat import Detector, Fex
from feat.utils import FEAT_EMOTION_COLUMNS
import pandas as pd 
import matplotlib.pyplot as plt

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
    
    for image in os.listdir(images_path):
        # loop heavily based on lab 1 exercise 4 solution
        curr_image_path = images_path + image
         
        # read current image and save as frame to run detection on
        frame = cv2.imread(curr_image_path)

        # call faces, landmark, emotion and au detection 
        faces = detector.detect_faces(frame) 
        landmarks = detector.detect_landmarks(frame, faces)
        emotions = detector.detect_emotions(frame, faces, landmarks)
        aus = detector.detect_aus(frame, landmarks)

        # access frame the only frame (image loaded)
        faces = faces[0]
        landmarks = landmarks[0]
        emotions = emotions[0]
        aus = aus[0]

        strongest_emotion = emotions.argmax(axis=1)
        for i, (face, top_emo) in enumerate(zip(faces, strongest_emotion)):
            (x0, y0, x1, y1, p) = face
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 3)
            if int(y0 - 10) > 0:
                cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            else:
                cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y1 + 20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            
            # create dictionary for row with current image name and face index
            au_row = {"file": image, "face": i }

            # iterate over aus
            for j, au_value in enumerate(aus[i]):
                # add AU index and values to dictionary
                au_row[f"AU{j + 1}"] = au_value
            
            # append the row to au_data list
            au_data.append(au_row)
            
        # save image with added bounding box and emotion label
        cv2.imwrite(output_path + image, frame)

    # create a DataFrame with AU activations
    au_df = pd.DataFrame(au_data)

    # save AU DataFrame to csv file
    au_df.to_csv(aus_path, index=False)

    # ----------------- Split data into positive and negative -----------------

    # filter out extensions from file names
    au_df['file'] = au_df['file'].apply(lambda x: os.path.splitext(x)[0])

    positive_data = []
    negative_data = []
    
    # sort by valence
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
    # -------------------------------------------------------------------------
    
    # ----------------- Calculate mean of each AU for positive and negative data and plot -----------------

    # go through negative_data.csv and positive_data.csv and calculate the mean of each AU
    # for each au, calculate the absolute difference between the mean of the positive and negative data
    # sort the AUs by the absolute difference
    # plot and save the sorted list of AUs

    abs_diff = {}

    # iterate over every AU column
    for col in negative_df.columns:

        # skip first two columns
        if col == "file" or col == "face":
            continue

        # calculate the absolute mean diff for the AU
        positive_mean = positive_df[col].mean()
        negative_mean = negative_df[col].mean()
        absolute_difference = abs(positive_mean - negative_mean)
        
        # save the absolute mean diff for the AU for plotting
        abs_diff[col] = absolute_difference

    # sort the AUs by the absolute mean diff (descending)
    abs_diff = dict(sorted(abs_diff.items(), key=lambda item: item[1], reverse=True))

    # create a DataFrame with AU names and their absolute mean diff values
    abs_diff_df = pd.DataFrame([abs_diff])
    abs_diff_df.to_csv('processed/abs_diff.csv', index=False)

    # plot all values
    plt.plot(list(abs_diff.keys()), list(abs_diff.values()), 'b.')

    # color the 6 highest mean diff AUs red
    top_6_aus = list(abs_diff.keys())[:6]
    top_6_values = list(abs_diff.values())[:6]
    plt.plot(top_6_aus, top_6_values, 'r.') 
    plt.xticks(rotation=45)
    plt.xlabel('AU')
    plt.ylabel('Absolute difference')
    plt.title('Absolute difference in AU means (pos. vs neg.)')
    plt.savefig('processed/au_visualization.png')

    print("Processing done")

main()