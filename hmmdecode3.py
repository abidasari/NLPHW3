#!/usr/bin/python
import codecs
import pickle
import sys
import time

start_time = time.time()
tags = {u'DI': 0, u'DD': 1, u'DA': 2, u'WW': 3, u'FF': 4, u'DT': 5, u'DR': 6, u'DP': 7, u'PR': 8, u'PP': 9, u'PT': 10, u'PX': 11, u'NC': 12, u'RG': 13, u'PD': 14, u'NP': 15, u'RN': 16, u'PI': 17, u'VA': 18, u'P0': 19, u'CC': 20, u'VM': 21, u'AO': 22, u'AQ': 23, u'VS': 24, u'ZZ': 25, u'CS': 26, u'II': 27, u'SP': 28, u'START': 29, u'END': 30}

Dic = pickle.load(open("emission.p", "rb"))
tags_count_arr = pickle.load(open("tagscountarr.p", "rb"))

emission = Dic

for word in Dic:# creating the emission matrix
    for val in range(31):
        emission[word][val] = Dic[word][val] / tags_count_arr[val]

transmission = pickle.load(open("transmission.p", "rb"))
all_the_tags = pickle.load(open("tags.p", "rb"))

outfile = open("hmmoutput.txt", "w")


basepath = str(sys.argv[1])

def get_the_tags(line_from_doc):# Do viterbi here
    sentence_as_list = line_from_doc.split(" ")
    sentence_length = len(sentence_as_list)
    viterbi = [ ]
    backpointer = [ ]
    first_viterbi = { }
    first_backpointer = { }
    for tag in all_the_tags:
        if tag == "START": continue
        if sentence_as_list[0] in emission:
            first_viterbi[ tag ] = transmission["START"][tags[tag]] * emission[sentence_as_list[0]][tags[tag]]
        else:
            first_viterbi[tag] = transmission["START"][tags[tag]]
        first_backpointer[tag] = "START"

    viterbi.append(first_viterbi)
    backpointer.append(first_backpointer)

    for index in range(1, sentence_length):
        this_viterbi = { }
        this_backpointer = { }
        prev_viterbi = viterbi[-1]
        for tag in all_the_tags:
            if tag == "START": continue
            if sentence_as_list[index] in emission:
                if emission[sentence_as_list[index]][tags[tag]] != 0:
                    best_previous = max(prev_viterbi.keys(), key=lambda prevtag: prev_viterbi[prevtag] * transmission[prevtag][tags[tag]] * emission[sentence_as_list[index]][tags[tag]])
                    this_viterbi[tag] = prev_viterbi[best_previous] * transmission[best_previous][tags[tag]] * emission[sentence_as_list[index]][tags[tag]]
                    this_backpointer[tag] = best_previous
                else:
                    best_previous = max(prev_viterbi.keys())
                    this_viterbi[tag] = 0
                    this_backpointer[tag] = best_previous
            else:
                best_previous = max(prev_viterbi.keys(), key=lambda prevtag: prev_viterbi[prevtag] * transmission[prevtag][tags[tag]] )
                this_viterbi[tag] = prev_viterbi[best_previous] * transmission[best_previous][tags[tag]]
                this_backpointer[tag] = best_previous
        viterbi.append(this_viterbi)
        backpointer.append(this_backpointer)

    prev_viterbi =  viterbi[-1]
    best_previous = max(prev_viterbi.keys(), key=lambda prevtag: prev_viterbi[prevtag] * transmission[prevtag][tags["END"]])
    # prob_tag_seq = prev_viterbi[best_previous] + transmission[best_previous][tags["END"]]
    best_tag_seq = ["END", best_previous]
    backpointer.reverse()
    current_best_tag = best_previous
    for bp in backpointer:
        best_tag_seq.append(bp[current_best_tag])
        current_best_tag = bp[current_best_tag]
    best_tag_seq.reverse()
    outstring = ""
    for index, value in enumerate(sentence_as_list):
        outstring = outstring + value + "/" + best_tag_seq[index+1] + " "
    # print(outstring)
    return outstring

with codecs.open(basepath, 'r', encoding='utf8') as f:
    for line in f:
        line = line[:-1]  # removing the \n line breaks
        output_line = get_the_tags(line)
        output_line = output_line[:-1]
        outfile.write(output_line)
        outfile.write("\n")

outfile.close()
print("Decoding done in: %s"  % (time.time() - start_time) )