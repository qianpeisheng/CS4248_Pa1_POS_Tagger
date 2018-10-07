# python3.5 build-tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import pandas as pd
import pickle

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

# row is the 1st word, col is the 2nd word
# row does not have </s>; col does not have <s>
def build_bigram_tag_count_table(save, train_tags):
    number_of_rows = len(train_tags) + 1 # add 1 for <s> in row and </s> in col
    number_of_cols = number_of_rows
    table_of_zeros = np.zeros((number_of_rows, number_of_cols), dtype = float)
    # the table will be converted to a probability table, thus float
    row_tags, col_tags = ['1 <s>'],[]
    for i in train_tags:
        row_tags.append('1 ' + i)
        col_tags.append('2 ' + i)
    col_tags.append('2 </s>')
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
    return df

# numbers are not exhaustive -- handle separately
# everything saved to small letters except NNP and NNPS
# <UNK> = words with count <= 1
# not smoothed
def build_tag_word_dict(save, train_tags):
    tag_word_dict = {}
    for tag in train_tags:
        tag_word_dict[tag] = {}
    for sentence in save:
        for word_tag_pair in sentence:
            word = ''
            # if word_tag_pair[1] == 'NNP' or word_tag_pair[1] == 'NNPS':
            #     word = word_tag_pair[0]
            # else:
            #     word = word_tag_pair[0].lower()
            word = word_tag_pair[0].lower()
            if word not in tag_word_dict[word_tag_pair[1]]:
                tag_word_dict[word_tag_pair[1]][word] = 1
            else:
                tag_word_dict[word_tag_pair[1]][word] += 1
    # transform all count == 1 to unknwon
    # deal with the case where <UNK> is too high for NNP and NNPS
    for key, value in tag_word_dict.items():
        new_word_count_dict = {'<UNK>':0}
        count_plus = 1
        if key == 'NNP' or key == 'NNPS':
            #count_plus = 0.5
            count_plus = 1
        for word, count in value.items():
            if count > 1:
                new_word_count_dict[word] = count
            else:
                new_word_count_dict['<UNK>'] += count_plus
        tag_word_dict[key] = new_word_count_dict
    #change counts to proba
    for key, value in tag_word_dict.items():
        sum_of_values = sum(value.values())
        for k, v in value.items():
            value[k] = v/sum_of_values
    return tag_word_dict

# implementation of the simple add one smoothing
def add_one_smoothing(df):
    def add_one(x):
            return x + 1
    return df.applymap(add_one)

def get_tag_proba_df(df):
    # index is string
    for index, row in df.iterrows():
        sum = row.sum()
        df.loc[index] = row.apply(lambda x: x/sum)
    return df

def save_model(tag_word_dict, tag_df):
    with open('model.pkl', 'wb') as f:
        pickle.dump([tag_word_dict, tag_df], f)
    print('saved')

def train_model(train_file, model_file):
    save = train_file_to_list(train_file)
    train_tags = get_train_tags(save)
    tag_word_dict = build_tag_word_dict(save, train_tags)
    tag_count_df = build_bigram_tag_count_table(save, train_tags)
    tag_count_df_add_one = add_one_smoothing(tag_count_df)
    proba_df = get_tag_proba_df(tag_count_df_add_one)
    print('proba_df', proba_df)
    save_model(tag_word_dict, proba_df)
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
