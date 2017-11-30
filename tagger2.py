from tagger1 import *

if __name__ == '__main__':
    if len(sys.argv) > 1:
        vecs = np.loadtxt(sys.argv[1])
        words, W2I = read_file(sys.argv[2], parse_vocab_reading)
        train_pos, dev_pos, W2I_pos, T2I_pos, I2T_pos = prepare_data("data/pos/train", "data/pos/dev", " ")
        train_ner, dev_ner, W2I_ner, T2I_ner, I2T_ner = prepare_data("data/ner/train", "data/ner/dev", "\t")
        dev_pos_losses, dev_pos_accs = train_on_dataset(train_pos, dev_pos, 30, 0.01, W2I, T2I_pos, I2T_pos, 50,
                                                        "encoding_pos.model", emdM=vecs)
        train_on_dataset(train_ner, dev_ner, 30, 0.01, W2I, T2I_ner, I2T_ner,
                         50, "encoding_ner.model", droput1=0.4, droput2=0.4, ner=True, emdM=vecs)
