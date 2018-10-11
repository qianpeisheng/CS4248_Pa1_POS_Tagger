# python3.5 run-tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import math
import numpy as np
import pickle
import time

def load_model(model_file):
    with open(model_file, 'rb') as f:
        return pickle.load(f)
    
# a list of sentences, each sentence is a list of words
def test_file_to_list(test_file):
    sentence_save = []
    with open(test_file) as infile:
        for line in infile:
            sentence_save.append(line.split())
    return sentence_save

class Tagger:
    def __init__(self, tag_matrix, tag_word_count_dict, vocab_set, list_of_tags):
        self.tag_matrix = tag_matrix
        self.tag_word_count_dict = tag_word_count_dict
        self.vocab_set = vocab_set
        self.list_of_tags = list_of_tags
        self.number_of_tags = len(list_of_tags)

    def get_pos(self, sentence):
        viterbi_table = []
        last_pos = '<s>'
        #last_col = np.zeros(self.number_of_tags) # log 1 = 0
        last_col = [(0, '<s>') for i in range(self.number_of_tags)]

        for word_index, word in enumerate(sentence):
            word = word#.lower()
            new_col = []
            if word in self.vocab_set:
                for index, tag in enumerate(self.list_of_tags):
                    log_p_w_t = -999999
                    tag_col = index # col append </s> in the end
                    if word not in self.tag_word_count_dict[tag]:
                        log_p_w_t = -999 # assume it is *not* an unknown of this tag
                    else:
                        log_p_w_t = math.log2(self.tag_word_count_dict[tag][word])
                    log_proba_list = []
                    for index, log_proba_tag_pair in enumerate(last_col):
                        tag_row = -1
                        if word_index == 0:
                            tag_row = 0
                        else:
                            tag_row = index + 1 # row 0 is <s>
                        log_p_t_t0 = math.log2(self.tag_matrix[tag_row][tag_col])
                        log_proba_list.append(log_p_t_t0 + log_proba_tag_pair[0])
                    max_log_proba = max(log_proba_list)
                    max_log_proba_index_pair = (max_log_proba  + log_p_w_t, log_proba_list.index(max_log_proba))
                    new_col.append(max_log_proba_index_pair)
            else:
                for index, tag in enumerate(self.list_of_tags):
                    log_p_w_t = -999999
                    tag_col = index 
                    # know that it will be unknown
                    unk_proba = self.tag_word_count_dict[tag]['<UNK>']
                    if word[0].isupper() and word != sentence[0]:
                        if tag == 'NNP' or tag == 'NNPS':
                            unk_proba *= 100 # guess a word with capital first letter not at the start is NNP/NNPS
                    # handle the case of the first word
                    # if len(sentence) > 1:
                    #     if word[0].isupper() and word == sentence[0] and sentence[1][0].isupper():
                    #         if tag == 'NNP' or tag == 'NNPS':
                    #         # print(sentence, ' sentence')
                    #             unk_proba *= 100
                    if unk_proba == 0:
                        log_p_w_t = -999
                    else:
                        log_p_w_t = math.log2(unk_proba)
                    log_proba_list = []
                    for index, log_proba_tag_pair in enumerate(last_col):
                        tag_row = -1
                        if word_index == 0:
                            tag_row = 0
                        else:
                            tag_row = index + 1 # row 0 is <s>
                        log_p_t_t0 = math.log2(self.tag_matrix[tag_row][tag_col])
                        log_proba_list.append(log_p_t_t0 + log_proba_tag_pair[0])
                    max_log_proba = max(log_proba_list) 
                    max_log_proba_index_pair = (max_log_proba + log_p_w_t, log_proba_list.index(max_log_proba))
                    new_col.append(max_log_proba_index_pair)
            viterbi_table.append(new_col)
            last_col = new_col
        # compute </s>
        second_last_col = viterbi_table[len(viterbi_table) - 1]
        last_row = [(second_last_col[i][0] + math.log2(self.tag_matrix[i][self.number_of_tags]), i) for i in range(self.number_of_tags)]#45 col is </s> 
        last_tag_index = max(last_row, key = lambda x: x[0])[1]
        #trace the tag from last
        reverse_answer_tag_index_list = [last_tag_index]
        reverse_v_table = viterbi_table[::-1]
        for col in reverse_v_table:
            answer = col[last_tag_index][1]
            reverse_answer_tag_index_list.append(answer)
            last_tag_index = answer
        answer_tag_index_list = reverse_answer_tag_index_list[::-1]
        return [self.list_of_tags[i] for i in answer_tag_index_list[1:]]# the 0th predicts for <s>
    
def tag_sentence(test_file, model_file, out_file):
    tag_matrix, tag_word_count_dict, vocab_set, list_of_tags = load_model(model_file)
    tagger = Tagger(tag_matrix, tag_word_count_dict, vocab_set, list_of_tags)
    sentences = test_file_to_list(test_file)
    answers = []
    start = time.time()
    count = 0
    for sentence in sentences:
        count += 1
        if count % 200 == 0:
            print(time.time() - start)
        #print(sentence)
        answers.append(tagger.get_pos(sentence))
        #break
    # mix sentence and tags
    out = []
    for sentence, answer in zip(sentences, answers):
        out_line = ''
        for word, tag in zip(sentence, answer):
            out_line += word + '/' + tag + ' '
        out.append(out_line)
    
    # write to file
    with open(out_file, 'w') as f:
        for i in out:
            f.write(i + '\n')
    # write your code here. You can add functions as well.
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
