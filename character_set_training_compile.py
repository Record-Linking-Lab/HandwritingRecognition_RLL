## Tyler Dial 06/20 RLL Computer Vision Team

# GOAL: To compile a set of training data for the BYU hwr model using iam, census, and other datasets
# This training data will organized by character sets 

import csv
import re

def classify_line(line):
    if re.match(r'^[a-zA-Z]+$', line):
        return 'az'
    elif re.match(r'^[a-zA-Z0-9]+$', line):
        return 'az09'
    else:
        return 'extascii'

def process_tsv_file(file_path):
    with open(file_path, 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            line = row[0]  # Assuming there's only one column in each row
            classification = classify_line(line)
            print(f'{line}: {classification}')

# Usage example
tsv_file_path = 'data.tsv'
process_tsv_file(tsv_file_path)

