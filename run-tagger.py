# python3.5 run-tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import math
import numpy as np
import pandas as pd
import pickle

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

# key is pos
def get_new_node(key, word, last_col, tag_word_dict, tag_df):
    p_w_t = 0
    if word in tag_word_dict[key].keys():
        p_w_t = tag_word_dict[key][word]
    else:
        p_w_t = tag_word_dict[key]['<UNK>']
    proba_list = []
    for item in last_col:
        #print(item)
        p_t_t = math.log2(tag_df.loc['1 ' + item[0]]['2 ' + key]) + item[1]
        proba_list.append((item[0], p_t_t))# save last pos
    max_tag_proba_pair = max(proba_list, key = lambda x: x[1])
    combined_proba = max_tag_proba_pair[1]
    if p_w_t > 0:
        combined_proba += math.log2(p_w_t)
    return (key, combined_proba, max_tag_proba_pair[0])


# implementation of viterbi algorithm
# unknown proba is used if it is not known through out the vocab
def get_pos(sentence, tag_word_dict, tag_df):
    #viterbi_table = np.zeros((len(sentence) + 1, 45))
    viterbi_table = []
    position = 0
    for word in sentence:
        word = word.lower()
        if len(viterbi_table) == 0:
            # it is the first word
            new_col = []
            in_vocab = False
            for key, value in tag_word_dict.items():
                p_w_t = 0
                if word in value.keys():
                    in_vocab = True
                    p_w_t = value[word]
                # else:
                #     p_w_t = value['<UNK>']
                p_t_s = tag_df.loc['1 <s>']['2 ' + key]
                proba = math.log2(p_t_s)
                if p_w_t > 0:
                    proba += math.log2(p_w_t)
                else:
                    proba += -99999
                new_col.append((key, proba, '<s>'))
            if not in_vocab:
                print('word not in vocab ', word)
                new_col = []
                for key, value in tag_word_dict.items():
                    p_w_t = value['<UNK>']
                    p_t_s = tag_df.loc['1 <s>']['2 ' + key]
                    proba = math.log2(p_t_s)
                    if p_w_t > 0:
                        proba += math.log2(p_w_t)
                new_col.append((key, proba, '<s>'))
            viterbi_table.append(new_col)
        else:
            last_col = viterbi_table[position - 1]
            new_col = []
            in_vocab = False
            answer_list = []
            for key, value in tag_word_dict.items():
                p_w_t = 0
                if word in tag_word_dict[key].keys():
                    in_vocab = True
                    p_w_t = tag_word_dict[key][word]
                # else:
                #     p_w_t = tag_word_dict[key]['<UNK>']
                proba_list = []
                for item in last_col:
                    #print(item)
                    p_t_t = math.log2(tag_df.loc['1 ' + item[0]]['2 ' + key]) + item[1]
                    proba_list.append((item[0], p_t_t))# save last pos
                max_tag_proba_pair = max(proba_list, key = lambda x: x[1])
                combined_proba = max_tag_proba_pair[1]
                if p_w_t > 0:
                    combined_proba += math.log2(p_w_t)
                else:
                    combined_proba += -99999
                new_col.append((key, combined_proba, max_tag_proba_pair[0]))
                # new_col.append(get_new_node(key, word, last_col, tag_word_dict, tag_df))
            if not in_vocab:
                new_col = []
                for key, value in tag_word_dict.items():
                    p_w_t = tag_word_dict[key]['<UNK>']
                    proba_list = []
                    for item in last_col:
                        #print(item)
                        p_t_t = math.log2(tag_df.loc['1 ' + item[0]]['2 ' + key]) + item[1]
                        proba_list.append((item[0], p_t_t))# save last pos
                    max_tag_proba_pair = max(proba_list, key = lambda x: x[1])
                    combined_proba = max_tag_proba_pair[1]
                    if p_w_t > 0:
                        combined_proba += math.log2(p_w_t)
                    new_col.append((key, combined_proba, max_tag_proba_pair[0]))                
            viterbi_table.append(new_col)
        position += 1
    # add one more column for </s>
    last_word = sentence[position - 1]
    last_col = viterbi_table[position - 1]
    new_col = []
    for item in last_col:
        p_t_s = tag_df.loc['1 ' + item[0]]['2 </s>']
        proba = math.log2(p_t_s) + item[1]
        new_col.append(('</s>', proba, item[0]))
    viterbi_table.append(new_col)
    print('v table 1', viterbi_table[1])
    print(len(viterbi_table[1]))
    answer = []
    v_table_reverse = viterbi_table[::-1]
    answer_reversed = []
    count = 0
    next = '</s>'
    for col in v_table_reverse:
        answer_reversed.append(next)
        next = max(col, key=lambda x: x[1])[2]
    # for col in viterbi_table:
    #     max_tag = max(col, key=lambda x: x[1])
    #     answer.append(max_tag)
    #print('answer', answer)
    #return answer
    print('answer r r', answer_reversed[::-1])
    return answer_reversed[::-1]

def tag_sentence(test_file, model_file, out_file):
    tag_word_dict, tag_df = load_model(model_file)
    #print('1 DT', tag_df.loc['1 DT'])
    print('question', tag_word_dict['NN']['question']) # CD number needs to change!
    print('question' in tag_word_dict['NNP'])
    print('UNK ', tag_word_dict['NNP']['<UNK>']) # UNKNOWN is not handled correctly!!
    #print('for' in tag_word_dict['PRP$'])
    #print(tag_word_dict['NNPS'])
    sentences = test_file_to_list(test_file)
    answers = []
    for sentence in sentences[3:4]:
        print(sentence)
        answers.append(get_pos(sentence, tag_word_dict, tag_df))
        break
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
