from tagger1 import *
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) > 1:
        vecs = np.loadtxt(sys.argv[1])
        rand_mat_start = np.random.rand(vecs.shape[1])
        rand_mat_end = np.random.rand(vecs.shape[1])
        vecs = np.vstack((vecs, rand_mat_start))
        vecs = np.vstack((vecs, rand_mat_end))
        words, W2I = read_file(sys.argv[2], parse_vocab_reading)

        train_pos, dev_pos, W2I_pos, T2I_pos, I2T_pos = prepare_data("data/pos/train", "data/pos/dev", " ",
                                                                     most_to_take=40000)
        train_ner, dev_ner, W2I_ner, T2I_ner, I2T_ner = prepare_data("data/ner/train", "data/ner/dev", "\t")
        capital_W2I = dict()
        start = len(W2I)
        vecs = vecs.tolist()
        for word in W2I_pos:
            if word not in W2I:
                W2I[word] = start
                rand_mat = np.random.rand(EMBEDDING_SIZE).tolist()
                vecs.append(rand_mat)
                start += 1
        vecs = np.array(vecs)
        dev_pos_losses, dev_pos_accs = train_on_dataset(train_pos, dev_pos, 64, 0.01, W2I, T2I_pos, I2T_pos, 50,
                                                        "encoding_pos.model", emdM=vecs, droput1=0.4, droput2=0.4)
        dev_ner_losses, dev_ner_accs = train_on_dataset(train_ner, dev_ner, 64, 0.01, W2I, T2I_ner, I2T_ner,
                                                        50, "encoding_ner.model", emdM=vecs)
