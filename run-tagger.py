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
            word = word.lower()
            new_col = []
            if word in self.vocab_set:
                #print('in !', word)
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
                #print('not', word)
                for index, tag in enumerate(self.list_of_tags):
                    log_p_w_t = -999999
                    tag_col = index 
                    # know that it will be unknown
                    unk_proba = self.tag_word_count_dict[tag]['<UNK>']
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
    print('for' in tagger.vocab_set)
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
