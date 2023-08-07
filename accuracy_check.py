### Tyler Dial 08/01/2023 Computer Vision RLL

## GOAL: Run the BYU model on the iam dataset and see how well it handles incorrect data
# First we have to fix the filepaths on the labeled dataset
# Then we want to look at attributes of the dataset
# Then we need to split the data into a train/test set in a way that reduces bias

# Data: /home/tyler97/fsl_groups/fslg_death/compute/datasets/iam/training_data/relative_iam_word_labels.tsv

import csv
import random
import sys
from collections import Counter
from collections import defaultdict

# I was getting a csv field size error so this code will set a larger field size limit
csv.field_size_limit(sys.maxsize)

# ## FIX FILE PATHS
# # This function will turn relative filepaths into full filepaths

# def modify_filepaths(input_file, base_path, output_file=None):
#     output_rows = []

#     with open(input_file, 'r', newline='') as tsv_file:
#         reader = csv.reader(tsv_file, delimiter='\t')
#         for row in reader:
#             # Assuming the first column contains relative filepaths
#             relative_filepath = row[0]
#             full_filepath = f"{base_path}/{relative_filepath}"
#             row[0] = full_filepath
#             output_rows.append(row)

#     # Writing the modified data to the specified output file or a new TSV file
#     if output_file:
#         with open(output_file, 'w', newline='') as tsv_output:
#             writer = csv.writer(tsv_output, delimiter='\t')
#             writer.writerows(output_rows)

#         print(f"Modified TSV file has been saved to: {output_file}")
#     else:
#         output_file = input_file.replace(".tsv", "_modified.tsv")
#         with open(output_file, 'w', newline='') as tsv_output:
#             writer = csv.writer(tsv_output, delimiter='\t')
#             writer.writerows(output_rows)

#         print(f"Modified TSV file has been created: {output_file}")

# # Example usage
# if __name__ == "__main__":
#     input_tsv_file = "/grphome/fslg_death/compute/datasets/iam/training_data/relative_iam_word_labels.tsv"  
#     base_path_variable = "/grphome/fslg_death/compute/datasets/iam"
#     output_modified_tsv = "/home/tyler97/fsl_groups/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam/fullpath_iam_word_labels.tsv"  # Replace with desired output location

#     modify_filepaths(input_tsv_file, base_path_variable, output_modified_tsv)



## ANALYZE DATA
# We need to check the number of observations (to make sure that it's the same as the input file),
# see the number of unique words, number of words that occur <20x, and % of words that occur >20x

def analyze_tsv(input_file):
    word_counts = Counter()
    total_words = 0

    with open(input_file, 'r', newline='') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            # Assuming the second column contains words
            word = row[1]
            word_counts[word] += 1
            total_words += 1

    num_words_less_than_20 = sum(1 for count in word_counts.values() if count < 20)
    num_words_more_than_20 = total_words - num_words_less_than_20

    percentage_words_more_than_20 = (num_words_more_than_20 / total_words) * 100

    print(f"Total number of observations (words): {total_words}")
    print(f"Number of unique words: {len(word_counts)}")
    print(f"Number of words that occur less than 20 times: {num_words_less_than_20}")
    print(f"Percentage of words that occur more than 20 times: {percentage_words_more_than_20:.2f}%")

# Example usage
if __name__ == "__main__":
    input_tsv_file = "/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam/fullpath_iam_word_labels.tsv"  # Replace with your TSV file path

    analyze_tsv(input_tsv_file)


## SPLIT THE DATA INTO TRAIN AND TEST SETS
# This will be done by organizing the dataset by unique words, dropping words that don't occur >20 times,
# shuffling within each group of unique words, randomly sampling 5% of each word, and assigning the remaining
# 95% of the obs as the train set

def split_data(input_file, train_ratio=0.95, output_train_file=None, output_test_file=None):
    word_groups = defaultdict(list)

    # Group observations by word in the second column
    with open(input_file, 'r', newline='') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            word = row[1]
            word_groups[word].append(row)

    # Drop words that occur less than 20 times
    word_groups = {word: observations for word, observations in word_groups.items() if len(observations) >= 20}

    # Shuffle and split each word group into train and test sets
    train_rows = []
    test_rows = []

    for word, observations in word_groups.items():
        random.shuffle(observations)
        num_test_rows = int(len(observations) * (1 - train_ratio))
        test_rows.extend(observations[:num_test_rows])
        train_rows.extend(observations[num_test_rows:])

    # Shuffle the train and test sets
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    # Write the training set to the specified output file or a new TSV file
    if output_train_file:
        with open(output_train_file, 'w', newline='') as tsv_output:
            writer = csv.writer(tsv_output, delimiter='\t')
            writer.writerows(train_rows)

        print(f"Training set has been saved to: {output_train_file}")
    else:
        output_train_file = input_file.replace(".tsv", "_train.tsv")
        with open(output_train_file, 'w', newline='') as tsv_output:
            writer = csv.writer(tsv_output, delimiter='\t')
            writer.writerows(train_rows)

        print(f"Training set has been saved to: {output_train_file}")

    # Write the test set to the specified output file or a new TSV file
    if output_test_file:
        with open(output_test_file, 'w', newline='') as tsv_output:
            writer = csv.writer(tsv_output, delimiter='\t')
            writer.writerows(test_rows)

        print(f"Test set has been saved to: {output_test_file}")
    else:
        output_test_file = input_file.replace(".tsv", "_test.tsv")
        with open(output_test_file, 'w', newline='') as tsv_output:
            writer = csv.writer(tsv_output, delimiter='\t')
            writer.writerows(test_rows)

        print(f"Test set has been saved to: {output_test_file}")

# Run it on iam dataset
if __name__ == "__main__":
    input_tsv_file = "/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam/fullpath_iam_word_labels.tsv"  # Replace with your TSV file path
    train_ratio = 0.95  # 95% of the data will be assigned to the training set

    output_train_tsv = "/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam/fullpath_iam_labels_train.tsv"  # Replace with desired training set output location
    output_test_tsv = "/grphome/fslg_census/compute/Machine_learning_models/BYU_handwriting_model/datasets_for_training_data/iam/fullpath_iam_labels_test.tsv"    # Replace with desired test set output location

    split_data(input_tsv_file, train_ratio, output_train_tsv, output_test_tsv)


## Tests to make sure everything worked properly
# Same number of output obs as input obs
# Do we have any lexicon where there are less than 20 obs
# What proportion of the words have more than 20 words
# If the majority of the lexicons have more than 20 words, drop those with less than 20
# shuffle and sample the test set to be 5% of each lexicon
# shuffle and sample the train set to be the remaining 95% of each lexicon