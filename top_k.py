import numpy as np
from Utils import *

def most_similar(word, k):
    global vecs
    global words
    global W2I
    word = vecs[W2I[word]]  # get the dog vector
    sims = vecs.dot(word)  # compute similarities
    most_similar_ids = sims.argsort()[-2:-k - 2:-1]
    return [(words[i], sims[i]) for i in most_similar_ids]

if __name__ == '__main__':
    vecs = np.loadtxt("data/wordVectors.txt")
    words, W2I = read_file("data/vocab.txt", parse_vocab_reading)
    vecs_norm = np.linalg.norm(vecs, axis=1)
    vecs_norm.shape = (vecs_norm.shape[0], 1)
    vecs /= vecs_norm
    print most_similar('dog', 5)
    print most_similar('england', 5)
    print most_similar('john', 5)
    print most_similar('explode', 5)
    print most_similar('office', 5)


