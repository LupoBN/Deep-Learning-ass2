import numpy as np
from Utils import *
import sys
def most_similar(word, k):
    global vecs
    global words
    global W2I
    word = vecs[W2I[word]]  # get the dog vector
    sims = vecs.dot(word)  # compute similarities
    most_similar_ids = sims.argsort()[-2:-k - 2:-1]
    return [(words[i], sims[i]) for i in most_similar_ids]

if __name__ == '__main__':
    vecs = np.loadtxt(sys.argv[1])
    words, W2I = read_file(sys.argv[2], parse_vocab_words_reading)
    vecs_norm = np.linalg.norm(vecs, axis=1)
    vecs_norm.shape = (vecs_norm.shape[0], 1)
    vecs /= vecs_norm
    print most_similar('dog', 5)
    print most_similar('england', 5)
    print most_similar('john', 5)
    print most_similar('explode', 5)
    print most_similar('office', 5)


