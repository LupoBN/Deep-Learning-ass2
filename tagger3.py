from tagger1 import *
from MLPNetwork import MLPNetowrkSubWords

if __name__ == '__main__':
    train_pos, dev_pos, W2I_pos, T2I_pos, I2T_pos = prepare_data("data/pos/train", "data/pos/dev", " ")
    train_ner, dev_ner, W2I_ner, T2I_ner, I2T_ner = prepare_data("data/ner/train", "data/ner/dev", "\t")
    sub_map_pos = sub_words_mapping(train_pos, len(W2I_pos))
    W2I_pos.update(sub_map_pos)
    sub_map_ner = sub_words_mapping(train_ner, len(W2I_ner))
    W2I_ner.update(sub_map_ner)

    dev_pos_losses, dev_pos_accs = train_on_dataset(train_pos, dev_pos, 64, 0.01, W2I_pos, T2I_pos, I2T_pos, 50,
                                                    "subwords_pretrained_pos.model",
                                                    network_constructor=MLPNetowrkSubWords)
    dev_ner_losses, dev_ner_accs = train_on_dataset(train_ner, dev_ner, 30, 0.01, W2I_ner, T2I_ner, I2T_ner, 50,
                                                    "subwords_pretrained_pos.model",
                                                    network_constructor=MLPNetowrkSubWords, ner=True)
