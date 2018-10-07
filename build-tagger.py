# python3.5 build-tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import pandas as pd


def tag_to_index(save):
    temp_dict = {}
    for sentence in save:
        for word_tag in sentence:
            if word_tag[1] not in temp_dict.keys():
                temp_dict = None
    return True
def load_one_line(line):
    line_split_by_space = line.split()
    word_tag_pair_list = []
    for item in line_split_by_space:
        item_split = item.split("/")
        if len(item_split) > 2:
            item_combine = item_split[0]
            for i in range(1, len(item_split) - 1):
                item_combine = item_combine + '/' + item_split[i]
            word_tag_pair_list.append([item_combine, item_split[i + 1]])
        else:
            word_tag_pair_list.append(item_split)
    return word_tag_pair_list

def get_train_tags(save):
    temp_list = []
    for sentence in save:
        for word_tag in sentence:
            if word_tag[1] not in temp_list:
                temp_list.append(word_tag[1])
    print('temp list', temp_list)
    print('number of tags ', len(temp_list))
    return temp_list

def train_file_to_list(train_file):
    word_tag_pair_save = []
    with open(train_file) as infile:
        for line in infile:
            word_tag_pair_save.append(load_one_line(line))
    # print('               ', len(word_tag_pair_save))
    return word_tag_pair_save

def build_bigram_tag_count_table(save, train_tags):
    number_of_rows = len(train_tags) + 2 # add 2 for <s> and </s>
    number_of_cols = number_of_rows
    table_of_zeros = np.zeros((number_of_rows, number_of_cols), dtype = int)
    train_tags_extended = ['<s>']
    train_tags_extended.extend(train_tags)
    train_tags_extended.append('</s>')
    row_tags, col_tags = [],[]
    for i in train_tags_extended:
        row_tags.append('1 ' + i)
        col_tags.append('2 ' + i)
    df = pd.DataFrame(table_of_zeros, columns = col_tags, index = row_tags)
    # row index: the first word, col index: the second word
    row_tag = None
    col_tag = None
    for sentence in save:
        max_index = len(sentence) - 1
        for word__tag_index_pair in enumerate(sentence):
            col_tag = '2 ' + word__tag_index_pair[1][1]
            if (word__tag_index_pair[0] == 0):
                row_tag = '1 <s>'
            else:
                row_tag = '1 ' + sentence[word__tag_index_pair[0] - 1][1]# e.g. ['In', 'IN'], without index
            df[col_tag][row_tag] += 1
        col_tag = '2 </s>'
        row_tag = '1 ' + sentence[max_index][1]
        df[col_tag][row_tag] += 1
    print(df)
    return df

# need to handle upper and lower case
# numbers are not exhaustive
# everything saved to small letters except NNP and NNPS
def build_tag_word_dict(save, train_tags):
    tag_word_dict = {}
    for tag in train_tags:
        tag_word_dict[tag] = {}
    for sentence in save:
        for word_tag_pair in sentence:
            word = ''
            if word_tag_pair[1] == 'NNP' or word_tag_pair[1] == 'NNPS':
                word = word_tag_pair[0]
            else:
                word = word_tag_pair[0].lower()
            if word not in tag_word_dict[word_tag_pair[1]]:
                tag_word_dict[word_tag_pair[1]][word] = 1
            else:
                tag_word_dict[word_tag_pair[1]][word] += 1
    print(tag_word_dict['NN'])
    return tag_word_dict

def train_model(train_file, model_file):
    save = train_file_to_list(train_file)
    train_tags = get_train_tags(save)
    tag_word_dict = build_tag_word_dict(save, train_tags)
    #tag_count_df = build_bigram_tag_count_table(save, train_tags)
    # write your code here. You can add functions as well.
    print('   Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
