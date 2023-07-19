## Tyler Dial 06/20 RLL Computer Vision Team

# GOAL: To compile a set of training data for the BYU hwr model using iam, census, and other datasets
# This training data will organized by character sets 

import pandas as pd
import re

## For processing .txt datasets like iam
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


## For processsing csv datasets like for_torben data
def process_csv(file_path):
    df_az = pd.DataFrame(columns=['Image Path', 'Transcription'])
    df_az09 = pd.DataFrame(columns=['Image Path', 'Transcription'])
    df_ext_ascii = pd.DataFrame(columns=['Image Path', 'Transcription'])

    df = pd.read_csv(file_path)

    for index, row in df.iterrows():
        image_path = row['Image Path']
        transcription = row['Transcription']

        # Match based on the character set
        if re.match(r'^[a-zA-Z]+$', transcription):
            df_az = df_az.append({'Image Path': image_path, 'Transcription': transcription}, ignore_index=True)
        elif re.match(r'^[a-zA-Z0-9\s]+$', transcription):
            df_az09 = df_az09.append({'Image Path': image_path, 'Transcription': transcription}, ignore_index=True)
        else:
            df_ext_ascii = df_ext_ascii.append({'Image Path': image_path, 'Transcription': transcription}, ignore_index=True)

    return df_az, df_az09, df_ext_ascii

# For processing tsv datasets like mexico_train
def process_tsv(file_path):
    df_az = pd.DataFrame(columns=['Image Path', 'Transcription'])
    df_az09 = pd.DataFrame(columns=['Image Path', 'Transcription'])
    df_ext_ascii = pd.DataFrame(columns=['Image Path', 'Transcription'])

    df = pd.read_csv(file_path, delimiter='\t')  # Specify the delimiter as '\t' for TSV files

    for index, row in df.iterrows():
        image_path = row['Image Path']
        transcription = row['Transcription']

        # Match based on the character set
        if re.match(r'^[a-zA-Z]+$', transcription):
            df_az = df_az.append({'Image Path': image_path, 'Transcription': transcription}, ignore_index=True)
        elif re.match(r'^[a-zA-Z0-9\s]+$', transcription):
            df_az09 = df_az09.append({'Image Path': image_path, 'Transcription': transcription}, ignore_index=True)
        else:
            df_ext_ascii = df_ext_ascii.append({'Image Path': image_path, 'Transcription': transcription}, ignore_index=True)

    return df_az, df_az09, df_ext_ascii



## Create datasets from the dataframes
def create_csvs_from_dataframes(file_path, output_path):
    # create the dataframes based on .txt or .csv file type
    if '.txt' in file_path:
        df_az, df_az09, df_ext_ascii = process_txt(file_path)
    elif '.tsv'in file_path:
        df_az, df_az09, df_ext_ascii = process_tsv(file_path)
    else:
        df_az, df_az09, df_ext_ascii = process_csv(file_path)

    # Print heads of dataframes to check work
    print('DF "az":')
    print(df_az.head())
    print('DF "az09":')
    print(df_az09.head())
    print('DF "ext_ascii":')
    print(df_ext_ascii.head())

    # Extract the file name from the file path
    csv_name = file_path.split('/')[-1].split('.')[0]

    # This is where we store results for a-z
    az_csv_name = csv_name + '_az'
    csv_path_az = f'{output_path}/{az_csv_name}.csv'

    # This is where we store results for a-z and 0-9
    az09_csv_name = csv_name + '_az09'
    csv_path_az09 = f'{output_path}/{az09_csv_name}.csv'

    # This is where we store results for extended ASCII
    ext_ascii_csv_name = csv_name + '_ext_ascii'
    csv_path_ext_ascii = f'{output_path}/{ext_ascii_csv_name}.csv'

    # Create CSV files
    df_az.to_csv(csv_path_az, index=False)
    df_az09.to_csv(csv_path_az09, index=False)
    df_ext_ascii.to_csv(csv_path_ext_ascii, index=False)

# Use functions starting with iam dataset
iam = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam/file_list_words_training.txt'
iam_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam'
create_csvs_from_dataframes(iam, iam_output_path)

# Create csvs from for_torben folder
feb1 = '/grphome/fslg_census/compute/projects/for_torben/training_data_feb1'
feb1_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/training_data_feb1'
create_csvs_from_dataframes(feb1, feb1_output_path)

feb24_age = '/grphome/fslg_census/compute/projects/for_torben/feb24_age_training_data.csv'
feb24_age_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/feb24_age.csv'
create_csvs_from_dataframes(feb24_age, feb24_age_output_path)

mar9_father_birth = '/grphome/fslg_census/compute/projects/for_torben/father_birthplaces_denmark/mar9_father_birth_training_data.csv'
mar9_father_birth_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/mar9_father_birth.csv'
create_csvs_from_dataframes(mar9_father_birth, mar9_father_birth_output_path)

new_training_data = '/grphome/fslg_census/compute/projects/for_torben/ages_only/new_training_data.csv'
new_training_data_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/new_training_data.csv'
create_csvs_from_dataframes(new_training_data, new_training_data_output_path)

feb15_race = '/grphome/fslg_census/compute/projects/for_torben/training_data_census_files/feb15_race_training_data'
feb15_race_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/feb15_race.csv'
create_csvs_from_dataframes(feb15_race, feb15_race_output_path)

mar9_relationship = '/grphome/fslg_census/compute/projects/for_torben/training_data_census_files/mar9_relationship_training_data.csv'
mar9_relationship_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/mar9_relationship.csv'
create_csvs_from_dataframes(mar9_relationship, mar9_relationship_output_path)

mar9_birthplace = '/grphome/fslg_census/compute/projects/for_torben/training_data_census_files/mar9_birthplace_training_data.csv'
mar9_birthplace_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/mar9_birthplace.csv'
create_csvs_from_dataframes(mar9_birthplace, mar9_birthplace_output_path)

mar9_father_birth_training = '/grphome/fslg_census/compute/projects/for_torben/training_data_census_files/mar9_father_birth_training_data.csv'
mar9_father_birth_training_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/mar9_father_birth_training.csv'
create_csvs_from_dataframes(mar9_father_birth_training, mar9_father_birth_training_output_path)

feb24_age_training = '/grphome/fslg_census/compute/projects/for_torben/training_data_census_files/feb24_age_training_data.csv'
feb24_age_training_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/feb24_age_training.csv'
create_csvs_from_dataframes(feb24_age_training, feb24_age_training_output_path)

mar9_relationship_denmark = '/grphome/fslg_census/compute/projects/for_torben/relationships_denmark/mar9_relationship_training_data.csv'
mar9_relationship_denmark_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/mar9_relationship_denmark.csv'
create_csvs_from_dataframes(mar9_relationship_denmark, mar9_relationship_denmark_output_path)

training1_output = '/grphome/fslg_census/compute/projects/for_torben/training_outputs/training1'
training1_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/training1.csv'
create_csvs_from_dataframes(training1_output, training1_output_path)

gender_training = '/grphome/fslg_census/compute/projects/for_torben/training_outputs/gender_training.csv'
gender_training_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/gender_training.csv'
create_csvs_from_dataframes(gender_training, gender_training_output_path)

mar9_birthplace_denmark = '/grphome/fslg_census/compute/projects/for_torben/birthplaces_denmark/mar9_birthplace_training_data.csv'
mar9_birthplace_denmark_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/mar9_birthplace_denmark.csv'
create_csvs_from_dataframes(mar9_birthplace_denmark, mar9_birthplace_denmark_output_path)

race_training_data_6 = '/grphome/fslg_census/compute/projects/for_torben/races_for_denmark/race_training_data_6.csv'
race_training_data_6_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/race_training_data_6.csv'
create_csvs_from_dataframes(race_training_data_6, race_training_data_6_output_path)

gender_training_denmark = '/grphome/fslg_census/compute/projects/for_torben/genders_for_denmark/gender_training.csv'
gender_training_denmark_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/for_torben/gender_training_denmark.csv'
create_csvs_from_dataframes(gender_training_denmark, gender_training_denmark_output_path)

# Handwriting datasets (found in the hwr folder)
fixed_t_n_d_surnames_val = '/grphome/fslg_census/compute/projects/hwr/data_labeled/fixed_transcriptions_names_dashed_surnames_20191025_val'
fixed_t_n_d_surnames_val_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/hwr/fixed_transcriptions_names_dashed_surnames_20191025_val.csv'
create_csvs_from_dataframes(fixed_t_n_d_surnames_val, fixed_t_n_d_surnames_val_output_path)

fixed_t_n_d_surnames_train = '/grphome/fslg_census/compute/projects/hwr/data_labeled/transcriptions_names_dashed_surnames_20191025_train'
fixed_t_n_d_surnames_train_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/hwr/transcriptions_names_dashed_surnames_20191025_train.csv'
create_csvs_from_dataframes(fixed_t_n_d_surnames_train, fixed_t_n_d_surnames_train_output_path)

fixed_t_n_d_surnames_test = '/grphome/fslg_census/compute/projects/hwr/data_labeled/fixed_transcriptions_names_dashed_surnames_20191025_test'
fixed_t_n_d_surnames_test_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/hwr/fixed_transcriptions_names_dashed_surnames_20191025_test.csv'
create_csvs_from_dataframes(fixed_t_n_d_surnames_test, fixed_t_n_d_surnames_test_output_path)

fixed_t_n_o_full_names_val = '/grphome/fslg_census/compute/projects/hwr/data_labeled/fixed_transcriptions_names_only_full_names_val'
fixed_t_n_o_full_names_val_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/hwr/fixed_transcriptions_names_only_full_names_val.csv'
create_csvs_from_dataframes(fixed_t_n_o_full_names_val, fixed_t_n_o_full_names_val_output_path)

fixed_t_n_o_full_names_train = '/grphome/fslg_census/compute/projects/hwr/data_labeled/fixed_transcriptions_names_only_full_names_train'
fixed_t_n_o_full_names_train_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/hwr/fixed_transcriptions_names_only_full_names_train.csv'
create_csvs_from_dataframes(fixed_t_n_o_full_names_train, fixed_t_n_o_full_names_train_output_path)

fixed_t_n_o_full_names_test = '/grphome/fslg_census/compute/projects/hwr/data_labeled/fixed_transcriptions_names_only_full_names_test'
fixed_t_n_o_full_names_test_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/hwr/fixed_transcriptions_names_only_full_names_test.csv'
create_csvs_from_dataframes(fixed_t_n_o_full_names_test, fixed_t_n_o_full_names_test_output_path)

# Segmentation Datasets
t_n_d_surnames_val_segmentation = '/grphome/fslg_census/compute/projects/segmentation/data_labeled/transcriptions_names_dashed_surnames_with_blanks_20191111_val'
t_n_d_surnames_val_segmentation_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/segmentation/transcriptions_names_dashed_surnames_with_blanks_20191111_val.csv'
create_csvs_from_dataframes(t_n_d_surnames_val_segmentation, t_n_d_surnames_val_segmentation_output_path)

t_n_d_surnames_randomized_segmentation = '/grphome/fslg_census/compute/projects/segmentation/data_labeled/transcriptions_names_dashed_surnames_with_blanks_20191111_randomized'
t_n_d_surnames_randomized_segmentation_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/segmentation/transcriptions_names_dashed_surnames_with_blanks_20191111_randomized.csv'
create_csvs_from_dataframes(t_n_d_surnames_randomized_segmentation, t_n_d_surnames_randomized_segmentation_output_path)

t_n_d_surnames_test_segmentation = '/grphome/fslg_census/compute/projects/segmentation/data_labeled/transcriptions_names_dashed_surnames_with_blanks_20191111_test'
t_n_d_surnames_test_segmentation_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/segmentation/transcriptions_names_dashed_surnames_with_blanks_20191111_test.csv'
create_csvs_from_dataframes(t_n_d_surnames_test_segmentation, t_n_d_surnames_test_segmentation_output_path)

t_n_d_surnames_train_segmentation = '/grphome/fslg_census/compute/projects/segmentation/data_labeled/transcriptions_names_dashed_surnames_with_blanks_20191111_train'
t_n_d_surnames_train_segmentation_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/segmentation/transcriptions_names_dashed_surnames_with_blanks_20191111_train.csv'
create_csvs_from_dataframes(t_n_d_surnames_train_segmentation, t_n_d_surnames_train_segmentation_output_path)

# Mexico Census Datasets
mexico_train = '/grphome/fslg_census/compute/projects/Mexico_Census/segments/mexico_train.tsv'
mexico_train_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/Mexico_Census/mexico_train.tsv'
create_csvs_from_dataframes(mexico_train, mexico_train_output_path)

mexico_test_3k_sampled_images = '/grphome/fslg_census/compute/projects/Mexico_Census/testing_sets_mexico/3k_sampled_images/3k_test.csv'
mexico_test_3k_sampled_images_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/Mexico_Census/3k_sampled_images/3k_test.csv'
create_csvs_from_dataframes(mexico_test_3k_sampled_images, mexico_test_3k_sampled_images_output_path)

mexico_train_3k_sampled_images = '/grphome/fslg_census/compute/projects/Mexico_Census/testing_sets_mexico/3k_sampled_images/3k_images_folder_jackson/3k_train.tsv'
mexico_train_3k_sampled_images_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/Mexico_Census/3k_sampled_images/3k_train.tsv'
create_csvs_from_dataframes(mexico_train_3k_sampled_images, mexico_train_3k_sampled_images_output_path)

mexico_validated_data = '/grphome/fslg_census/compute/projects/Mexico_Census/Mexican-Census/6-reverse-indexing/folder_7/folder_4_validated_data.csv'
mexico_validated_data_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/Mexico_Census/validated_data.csv'
create_csvs_from_dataframes(mexico_validated_data, mexico_validated_data_output_path)

mexico_census_f7_validated = '/grphome/fslg_census/compute/projects/Mexico_Census/output/ri_raw_out/f_7/mexico_census_f7_1:7_validated.csv'
mexico_census_f7_validated_output_path = '/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/Mexico_Census/mexico_census_f7_1_7_validated.csv'
create_csvs_from_dataframes(mexico_census_f7_validated, mexico_census_f7_validated_output_path)