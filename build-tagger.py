# python3.5 build-tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import pickle

# work flow
# read in text, preprocess, save in memory
# go though text twice, 1st time, get the list of tags, tag_word_count_dict, vocab_list
# 2nd time, build the bigram tag count matrix, tag_word_count_dict
# modify the stats:
# tag count matrix to smoothed tag_probability matirx
# tag_word_count_dict to change count = 1 to <UNK>
# vocab_list delete unknown words
# save

class Tagger:
    def __init__(self, word_tag_pair_save, model_file):
        self.word_tag_pair_save = word_tag_pair_save
        self.tag_matrix = None
        self.list_of_tags = []
        self.vocab_set = set()
        self.tag_word_count_dict = {}
        self.number_of_tags = 45 # hard code the number of tags, to speed up
        self.model_file = model_file

    # fill in the tag list, original tag word count dict, vocab list
    def get_tags_and_vocab(self):
        for sentence in self.word_tag_pair_save:
            is_first = True
            for word_tag_pair in sentence:
                word = word_tag_pair[0]#.lower() # lower case
                # further modify the cap of first letter
                # if len(sentence) > 1:
                #     second_word = sentence[1][0]
                #     if not second_word[0].isupper():
                #         ttt = 1
                        # lower the first word
                        #word = word.lower()
                tag = word_tag_pair[1]
                # if tag != 'NNP' and tag != 'NNPS' and is_first:
                #     word = word.lower()
                # is_first = False
                if tag not in self.list_of_tags:
                    self.list_of_tags.append(tag)
                    self.tag_word_count_dict[tag] = {}   
                self.vocab_set.add(word)
                if word in self.tag_word_count_dict[tag]:
                    self.tag_word_count_dict[tag][word] += 1
                else:
                    self.tag_word_count_dict[tag][word] = 1
        self.number_of_tags = len(self.list_of_tags)
    
    # count known and calculate proba
    def set_unknown(self):
        vocab_unknown = set()
        for word_dict in self.tag_word_count_dict.values():
            number_of_unknown = 0
            unknown_word_set = set()
            for word, count in word_dict.items():
                if count == 1:
                    number_of_unknown += 1
                    unknown_word_set.add(word)
            word_dict['<UNK>'] = number_of_unknown
            vocab_unknown = vocab_unknown.union(unknown_word_set)
            for word in unknown_word_set:
                word_dict.pop(word, None)
                #self.vocab_set.discard(word)
            count_sum = sum(word_dict.values())
            type_count = len(word_dict.keys())
            smooth_ratio = 0
            for word, count in word_dict.items():
                # the idea is that the higher ratio the diversity, the more likely the unknown
                if (word_dict['<UNK>'] == 0):
                    # not smoothed, because unknwon will probably not appear
                    word_dict[word] = count/count_sum
                else:
                    if word == '<UNK>':
                        word_dict[word] = (count + type_count * smooth_ratio)/ (count_sum + type_count * smooth_ratio)
                    else:
                        word_dict[word] = count/ (count_sum + type_count * smooth_ratio)
        for word in vocab_unknown:
            in_dict = False
            for word_dict in self.tag_word_count_dict.values():
                if word in word_dict:
                    in_dict = True
                    break
            if not in_dict:
                self.vocab_set.discard(word)

    
    # matrix[t1][t2], t1 (row 0) starts with '<s>', t2 (column 45) ends with '</s>'
    # also compute proba
    def get_tag_matrix(self):
        number_of_rows = len(self.list_of_tags) + 1 # add 1 for <s> in row and </s> in col
        number_of_cols = number_of_rows
        self.tag_matrix = np.zeros((number_of_rows, number_of_cols), dtype = float)
        tag_col = self.list_of_tags.copy()
        tag_col.append('</s>')
        tag_row = ['<s>']
        tag_row.extend(self.list_of_tags)
        print(tag_col, tag_row)
        row_dict = {k : tag_row.index(k) for k in tag_row}
        col_dict = {k : tag_col.index(k) for k in tag_col}
        for sentence in self.word_tag_pair_save:
            max_index = len(sentence) - 1
            first = 0
            second = 1
            # a sentence has at least 1 word
            first_tag = '<s>'
            second_tag = sentence[first][1]
            first_index = row_dict[first_tag]
            second_index = col_dict[second_tag]
            self.tag_matrix[first_index][second_index] += 1
            while second < max_index:
                first_tag = sentence[first][1]
                second_tag = sentence[second][1]
                first_index = row_dict[first_tag]
                second_index = col_dict[second_tag]
                self.tag_matrix[first_index][second_index] += 1
                first += 1
                second += 1
            if max_index == 0:
                # there is only 1 word
                second = 0
            first_tag = sentence[second][1]
            second_tag = '</s>'
            first_index = row_dict[first_tag]
            second_index = col_dict[second_tag]
            self.tag_matrix[first_index][second_index] += 1

    # implementation of the simple add one smoothing
    def add_one_smoothing(self):
        for row in range(self.number_of_tags + 1):
            for col in range(self.number_of_tags + 1):
                self.tag_matrix[row][col] += 1
    
    # implementation of Witten-Bell Smoothing
    # V is very small
    def wb_smoothing(self):
        for row in range(self.number_of_tags + 1):
            T = 0
            Cw0 = 0
            V = self.number_of_tags + 1
            for col in range(self.number_of_tags + 1):
                if self.tag_matrix[row][col] != 0:
                    T += 1
                    Cw0 += self.tag_matrix[row][col] # no need to add 0
            for col in range(self.number_of_tags + 1):
                if self.tag_matrix[row][col] != 0:
                    self.tag_matrix[row][col] /= (Cw0 + T)
                else:
                    self.tag_matrix[row][col] =  T / ((V-T)*(Cw0+T))
    
    # implementation of interpolation
    
    def get_tag_proba(self):
        for row in range(self.number_of_tags + 1):
            count_sum = sum(self.tag_matrix[row])
            for col in range(self.number_of_tags + 1):
                self.tag_matrix[row][col] /= count_sum
        
    def save(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump([self.tag_matrix, self.tag_word_count_dict, self.vocab_set, self.list_of_tags], f)
        print('saved')
        
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

def train_file_to_list(train_file):
    word_tag_pair_save = []
    with open(train_file) as infile:
        for line in infile:
            word_tag_pair_save.append(load_one_line(line))
    return word_tag_pair_save

# row is the 1st word, col is the 2nd word
# row does not have </s>; col does not have <s>

# numbers are not exhaustive -- handle separately
# everything saved to small letters except NNP and NNPS
# <UNK> = words with count <= 1
# not smoothed

def train_model(train_file, model_file):
    word_tag_pair_save = train_file_to_list(train_file)
    tagger = Tagger(word_tag_pair_save, model_file)
    tagger.get_tags_and_vocab()
    tagger.set_unknown()
    tagger.get_tag_matrix()
    # tagger.add_one_smoothing()
    # tagger.get_tag_proba()
    tagger.wb_smoothing()
    tagger.save()
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
