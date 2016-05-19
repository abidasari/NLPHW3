#!/usr/bin/python
import codecs
import sys
import pickle
import time

start_time = time.time()

basepath = str(sys.argv[1])

vocab_count = 0

tags = {u'DI': 0, u'DD': 1, u'DA': 2, u'WW': 3, u'FF': 4, u'DT': 5, u'DR': 6, u'DP': 7, u'PR': 8, u'PP': 9, u'PT': 10, u'PX': 11, u'NC': 12, u'RG': 13, u'PD': 14, u'NP': 15, u'RN': 16, u'PI': 17, u'VA': 18, u'P0': 19, u'CC': 20, u'VM': 21, u'AO': 22, u'AQ': 23, u'VS': 24, u'ZZ': 25, u'CS': 26, u'II': 27, u'SP': 28, u'START': 29, u'END': 30}

tags_counts = {u'DI': 0, u'DD': 0, u'DA': 0, u'WW': 0, u'FF': 0, u'DT': 0, u'DR': 0, u'DP': 0, u'PR': 0, u'PP': 0, u'PT': 0, u'PX': 0, u'NC': 0, u'RG': 0, u'PD': 0, u'NP': 0, u'RN': 0, u'PI': 0, u'VA': 0, u'P0': 0, u'CC': 0, u'VM': 0, u'AO': 0, u'AQ': 0, u'VS': 0, u'ZZ': 0, u'CS': 0, u'II': 0, u'SP': 0, u'START': 0, u'END': 0}

tags_counts_arr = [0] * 31

transmission = {
    u'DI': [1] * 31, u'DD': [1] * 31, u'DA': [1] * 31, u'WW': [1] * 31, u'FF': [1] * 31, u'DT': [1] * 31, u'DR': [1] * 31, u'DP': [1] * 31, u'PR': [1] * 31, u'PP': [1] * 31, u'PT': [1] * 31,
    u'PX': [1] * 31, u'NC': [1] * 31, u'RG': [1] * 31, u'PD': [1] * 31, u'NP': [1] * 31, u'RN': [1] * 31, u'PI': [1] * 31, u'VA': [1] * 31, u'P0': [1] * 31, u'CC': [1] * 31,
    u'VM': [1] * 31, u'AO': [1] * 31, u'AQ': [1] * 31, u'VS': [1] * 31, u'ZZ': [1] * 31, u'CS': [1] * 31, u'II': [1] * 31, u'SP': [1] * 31, u'START': [1] * 31, u'END': [1] * 31}


tags_words = []# list of all tag word pairs in thr training data

words = [] # list of all words in the training data

Dic = {}

def add_to_dic(word ,tag):
    if(word in Dic):
        Dic[word][tags[tag]] += 1
    else:
        Dic[word] = [0] * 31
        Dic[word][tags[tag]] += 1

def sent_to_wordtagpairs(sentence):
    output = []
    sent_as_list = sentence.split(' ')
    for pair in sent_as_list:
        k = pair.rfind("/")
        word = pair[:k]
        tag = pair[k+1:]
        tags_counts[tag] = tags_counts[tag] + 1
        add_to_dic(word,tag)
        output.append((tag,word))
    return output

with codecs.open(basepath, 'r', encoding='utf8') as f:
    for line in f:
        line = line[:-1]#removing the \n line breaks
        tags_words.append(("START", "START"))
        tags_words.extend(sent_to_wordtagpairs(line))
        tags_words.append(("END", "END"))

emission = Dic

dict_tags = [tag for (tag, word) in tags_words] # only tags in order

all_the_tags = set(dict_tags)

for index, val in enumerate(dict_tags, start=0):
    prev_tag = dict_tags[index-1]
    transmission[prev_tag][tags[val]] += 1

for val in transmission:
    row_count = sum(transmission[val])
    # print(transmission["DI"][tags["DI"]])
    for int in range(31):
        transmission[val][int] /= row_count


for key, value in tags_counts.items():
    tags_counts_arr[tags[key]] = tags_counts[key]
tags_counts_arr[tags["START"]] = 1
tags_counts_arr[tags["END"]] = 1

pickle.dump(emission, open("emission.p", "wb"))
pickle.dump(transmission, open("transmission.p", "wb"))
pickle.dump(all_the_tags, open("tags.p", "wb"))
pickle.dump(tags_counts_arr, open("tagscountarr.p", "wb"))

print("Learning done in: %s"  % (time.time() - start_time) )