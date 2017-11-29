import copy
from collections import Counter


def parse_tag_reading(lines, seperator):
    words = list()
    labels = list()
    sentence = list()
    sentence_labels = list()
    for line in lines:
        if line != '':
            words_labels = line.rsplit(seperator, 1)
            sentence.append(words_labels[0])
            sentence_labels.append(words_labels[1])
        else:
            sentence = ["^^^^^", "^^^^^"] + sentence + ["$$$$$", "$$$$$"]
            sentence_labels = ["Start-", "Start-"] + sentence_labels + ["End-", "End-"]
            words.append(copy.deepcopy(sentence))
            labels.append(copy.deepcopy(sentence_labels))
            sentence = list()
            sentence_labels = list()
    return words, labels

def parse_vocab_reading(lines, seperator=None):
    words = [line for line in lines]
    W2I = {key: value for value, key in enumerate(words)}
    return words, W2I


def read_file(file_name, parse_func, seperator=None):
    file = open(file_name, 'r')
    lines = file.read().splitlines()
    file.close()
    return parse_func(lines, seperator)


def write_file(file_name, content, parse_func, seperator):
    file = open(file_name, 'w')
    file.write(parse_func(content, seperator))
    file.close()


def create_mapping(data, frequency_for_mapping=0, ignore_elements= None):
    count = count_uniques(data)
    possibles = set([x if count[x] > frequency_for_mapping else "UUUNKKK" for x in count])
    if ignore_elements != None:
        possibles = possibles.difference(ignore_elements)
    return {f: i for i, f in enumerate(list(sorted(possibles)))}


def count_uniques(sentences):
    fc = Counter()
    for words in sentences:
        fc.update(words)
    return fc
