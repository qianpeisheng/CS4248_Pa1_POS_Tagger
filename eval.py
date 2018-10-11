# python3.5 accuracy.py <output_file_absolute_path> <reference_file_absolute_path>
# make no changes in this file

import os
import sys


if __name__ == "__main__":
    out_file = sys.argv[1]
    reader = open(out_file)
    out_lines = reader.readlines()
    reader.close()

    ref_file = sys.argv[2]
    reader = open(ref_file)
    ref_lines = reader.readlines()
    reader.close()

    if len(out_lines) != len(ref_lines):
        print('Error: No. of lines in output file and reference file do not match.')
        exit(0)

    total_tags = 0
    matched_tags = 0
    error_dict = {}
    for i in range(0, len(out_lines)):
        cur_out_line = out_lines[i].strip()
        cur_out_tags = cur_out_line.split(' ')
        cur_ref_line = ref_lines[i].strip()
        cur_ref_tags = cur_ref_line.split(' ')
        total_tags += len(cur_ref_tags)
        
        for j in range(0, len(cur_ref_tags)):
            if cur_out_tags[j] == cur_ref_tags[j]:
                matched_tags += 1
            # debug 
            else:
                #print (out_lines[i])
                tag_0 = cur_out_tags[j].split('/')[1]
                tag_1 = cur_ref_tags[j].split('/')[1]
                if tag_0 not in error_dict:
                    error_dict[tag_0] = {}
                elif tag_1 not in error_dict[tag_0]:
                    error_dict[tag_0][tag_1] = 1
                else:
                    error_dict[tag_0][tag_1] += 1
                if tag_0 == 'NNP' and tag_1 == 'NN':
                    print(cur_out_line)
                    print(' out ', cur_out_tags[j], ' ref ', cur_ref_tags[j])
                #print('out ', cur_out_tags[j], ' ref ', cur_ref_tags[j])
    error_sum_count = 0
    for k, v in error_dict.items():
        for k1, v1 in v.items():
            error_sum_count +=v1
            if v1 >= 10:
                print(k, k1, v1)
    print(error_sum_count)

    print("Accuracy=", float(matched_tags) / total_tags)
