#!/usr/bin/python
import codecs
import sys
import pickle
import time

start_time = time.time() #To log the learning time

basepath = str(sys.argv[1]) #Path of the training data from the command line

# All the tags in the grammar and their index association
tags = {u'DI': 0, u'DD': 1, u'DA': 2, u'WW': 3, u'FF': 4, u'DT': 5, u'DR': 6, u'DP': 7, u'PR': 8, u'PP': 9, u'PT': 10, u'PX': 11, u'NC': 12, u'RG': 13, u'PD': 14, u'NP': 15, u'RN': 16, u'PI': 17, u'VA': 18, u'P0': 19, u'CC': 20, u'VM': 21, u'AO': 22, u'AQ': 23, u'VS': 24, u'ZZ': 25, u'CS': 26, u'II': 27, u'SP': 28, u'START': 29, u'END': 30}

#Another DS to keep track of the number of occurrences of each tag. Could be optimized.
tags_counts = {u'DI': 0, u'DD': 0, u'DA': 0, u'WW': 0, u'FF': 0, u'DT': 0, u'DR': 0, u'DP': 0, u'PR': 0, u'PP': 0, u'PT': 0, u'PX': 0, u'NC': 0, u'RG': 0, u'PD': 0, u'NP': 0, u'RN': 0, u'PI': 0, u'VA': 0, u'P0': 0, u'CC': 0, u'VM': 0, u'AO': 0, u'AQ': 0, u'VS': 0, u'ZZ': 0, u'CS': 0, u'II': 0, u'SP': 0, u'START': 0, u'END': 0}

tags_counts_arr = [0] * 31

# Data Structure to hold the transition Matrix [1] * 31 for smoothing
transition = {
    u'DI': [1] * 31, u'DD': [1] * 31, u'DA': [1] * 31, u'WW': [1] * 31, u'FF': [1] * 31, u'DT': [1] * 31, u'DR': [1] * 31, u'DP': [1] * 31, u'PR': [1] * 31, u'PP': [1] * 31, u'PT': [1] * 31,
    u'PX': [1] * 31, u'NC': [1] * 31, u'RG': [1] * 31, u'PD': [1] * 31, u'NP': [1] * 31, u'RN': [1] * 31, u'PI': [1] * 31, u'VA': [1] * 31, u'P0': [1] * 31, u'CC': [1] * 31,
    u'VM': [1] * 31, u'AO': [1] * 31, u'AQ': [1] * 31, u'VS': [1] * 31, u'ZZ': [1] * 31, u'CS': [1] * 31, u'II': [1] * 31, u'SP': [1] * 31, u'START': [1] * 31, u'END': [1] * 31}


tags_words = []# list of all tag word pairs in thr training data

words = [] # list of all words in the training data

Dic = {} # Dictionary will keep track of the count of occurrences too

def add_to_dic(word ,tag): # function will add a word tag pair to the dictionary
    if(word in Dic):
        Dic[word][tags[tag]] += 1
    else:
        Dic[word] = [0] * 31
        Dic[word][tags[tag]] += 1

def sent_to_wordtagpairs(sentence): # Converts a sentence to word tag pairs
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

with codecs.open(basepath, 'r', encoding='utf8') as f: # Read file in UTF-8 line-wise. Append START and END tags
    for line in f:
        line = line[:-1]#removing the \n line breaks
        tags_words.append(("START", "START"))
        tags_words.extend(sent_to_wordtagpairs(line))
        tags_words.append(("END", "END"))

emission = Dic # for the emission matrix

dict_tags = [tag for (tag, word) in tags_words] # Contains only tags in order

all_the_tags = set(dict_tags) # contains only the tags. Doing this because the initial learning and testing files were on different directories

for index, val in enumerate(dict_tags, start=0): # populating the transition matrix
    prev_tag = dict_tags[index-1]
    transition[prev_tag][tags[val]] += 1

for val in transition: # Calculating the transition probabilities
    row_count = sum(transition[val])
    for int in range(31):
        transition[val][int] /= row_count


for key, value in tags_counts.items():
    tags_counts_arr[tags[key]] = tags_counts[key]
tags_counts_arr[tags["START"]] = 1
tags_counts_arr[tags["END"]] = 1

pickle.dump(emission, open("emission.p", "wb"))
pickle.dump(transition, open("transition.p", "wb"))
pickle.dump(all_the_tags, open("tags.p", "wb"))
pickle.dump(tags_counts_arr, open("tagscountarr.p", "wb"))

print("Learning done in: %s"  % (time.time() - start_time) )

start_time = time.time()

Dic = pickle.load(open("emission.p", "rb"))
tags_count_arr = pickle.load(open("tagscountarr.p", "rb"))

emission = Dic

for word in Dic:# creating the emission matrix
    for val in range(31):
        emission[word][val] = Dic[word][val] / tags_count_arr[val]

transition = pickle.load(open("transition.p", "rb"))
all_the_tags = pickle.load(open("tags.p", "rb"))

outfile = open("hmmoutput.txt", "w")


basepath = str(sys.argv[1])

def get_the_tags(line_from_doc):# Viterbi Algorithm
    sentence_as_list = line_from_doc.split(" ")
    sentence_length = len(sentence_as_list)
    viterbi = [ ]
    backpointer = [ ]
    first_viterbi = { }
    first_backpointer = { }
    for tag in all_the_tags:
        if tag == "START": continue
        if sentence_as_list[0] in emission:
            first_viterbi[ tag ] = transition["START"][tags[tag]] * emission[sentence_as_list[0]][tags[tag]]
        else:
            first_viterbi[tag] = transition["START"][tags[tag]]
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
                    best_previous = max(prev_viterbi.keys(), key=lambda prevtag: prev_viterbi[prevtag] * transition[prevtag][tags[tag]] * emission[sentence_as_list[index]][tags[tag]])
                    this_viterbi[tag] = prev_viterbi[best_previous] * transition[best_previous][tags[tag]] * emission[sentence      _as_list[index]][tags[tag]]
                    this_backpointer[tag] = best_previous
                else:
                    best_previous = max(prev_viterbi.keys())
                    this_viterbi[tag] = 0
                    this_backpointer[tag] = best_previous
            else:
                best_previous = max(prev_viterbi.keys(), key=lambda prevtag: prev_viterbi[prevtag] * transition[prevtag][tags[tag]] )
                this_viterbi[tag] = prev_viterbi[best_previous] * transition[best_previous][tags[tag]]
                this_backpointer[tag] = best_previous
        viterbi.append(this_viterbi)
        backpointer.append(this_backpointer)

    prev_viterbi =  viterbi[-1]
    best_previous = max(prev_viterbi.keys(), key=lambda prevtag: prev_viterbi[prevtag] * transition[prevtag][tags["END"]])
    best_tag_seq = ["END", best_previous]
    backpointer.reverse()
    current_best_tag = best_previous
    for bp in backpointer:# Following Backpointers
        best_tag_seq.append(bp[current_best_tag])
        current_best_tag = bp[current_best_tag]
    best_tag_seq.reverse()
    outstring = ""
    for index, value in enumerate(sentence_as_list):
        outstring = outstring + value + "/" + best_tag_seq[index+1] + " "
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