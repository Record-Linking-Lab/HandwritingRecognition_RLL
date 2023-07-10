## Tyler Dial 06/20 RLL Computer Vision Team

# GOAL: To compile a set of training data for the BYU hwr model using iam, census, and other datasets
# This training data will organized by character sets 

import pandas as pd
import re

def process_txt(file_path):
    df_az = pd.DataFrame(columns=['Image Path', 'Transcription'])
    df_az09 = pd.DataFrame(columns=['Image Path', 'Transcription'])
    df_ext_ascii = pd.DataFrame(columns=['Image Path', 'Transcription'])

    with open(file_path, 'r') as txt_file:
        lines = txt_file.readlines()
        
        for line in lines:
            # Strip spaces, assign image path and transcription variable
            values = line.strip().split(' ')
            image_path = values[1]
            transcription = ' '.join(values[2:])
            # Match based on the character set
            if re.match(r'^[a-zA-Z]+$', transcription):
                df_az = df_az._append({'Image Path': image_path, 'Transcription': transcription}, ignore_index=True)
            elif re.match(r'^[a-zA-Z0-9\s]+$', transcription):
                df_az09 = df_az09._append({'Image Path': image_path, 'Transcription': transcription}, ignore_index=True)
            else:
                df_ext_ascii = df_ext_ascii._append({'Image Path': image_path, 'Transcription': transcription}, ignore_index=True)
    
    return df_az, df_az09, df_ext_ascii

# Usage example
txt_file_path = '/home/tyler97/fsl_groups/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam/file_list_words_training.txt'
df_az, df_az09, df_ext_ascii = process_txt(txt_file_path)

# Print the heads of the dataframes to check work
print('DF "az":')
print(df_az.head())

print('DF "az09":')
print(df_az09.head())

print('DF "ext_ascii":')
print(df_ext_ascii.head())

# Create csv's as output for training data based on character sets
az_file_path = '/home/tyler97/fsl_groups/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam_az.csv'
df_az.to_csv(az_file_path, index=False)

az09_file_path = '/home/tyler97/fsl_groups/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam_az09.csv'
df_az09.to_csv(az09_file_path, index=False)

ext_ascii_file_path = '/home/tyler97/fsl_groups/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam_ext_ascii.csv'
df_ext_ascii.to_csv(ext_ascii_file_path, index=False)