from Utils import *
from Helpers import *
from MLPNetwork import *
import sys

EMBEDDING_SIZE = 50


def train_on_dataset(train, dev, hidden_layer, W2I, T2I, I2T, num_of_iterations, save_file, dropout1=0.0,
                     dropout2=0.0, emdM=None, network_constructor=MLPNetwork):
    m = dy.ParameterCollection()
    # create network
    network = network_constructor(m, [EMBEDDING_SIZE * 5, hidden_layer, hidden_layer, len(T2I)], len(W2I), W2I, T2I,
                                  I2T, emdM)
    # create trainer
    trainer = dy.AdamTrainer(m)
    dev_losses, dev_accs = train_model(train, dev, network, trainer,
                                       num_of_iterations, save_file, dropout1, dropout2)
    return dev_losses, dev_accs


def test_model(dev, model_file, hidden_layer, W2I, T2I, I2T):
    m = dy.ParameterCollection()

    # create network
    network = MLPNetwork(m, [EMBEDDING_SIZE * 5, hidden_layer, hidden_layer, len(T2I)], len(W2I), W2I, T2I, I2T)

    network.load_model(model_file)

    dev_loss, dev_acc = test_data(dev, network)
    print dev_loss, dev_acc


def prepare_pretrained(W2I_train):
    vecs = None
    if "-pt" in sys.argv:
        W2I = read_file(sys.argv[-1], parse_vocab_reading)

        vecs = np.loadtxt(sys.argv[-2])
        vecs = vecs.tolist()
        for word in W2I_train:
            if word not in W2I:
                W2I[word] = len(W2I)
                rand_mat = np.random.rand(EMBEDDING_SIZE).tolist()
                vecs.append(rand_mat)
        vecs = np.array(vecs)
        W2I_train = W2I
    return vecs, W2I_train


def prepare_data(train_data_file, dev_data_file, separator, lower=False, most_to_take=15000):
    my_network = MLPNetwork

    train, labels = read_file(train_data_file, parse_tag_reading, separator, lower)
    dev, dev_labels = read_file(dev_data_file, parse_tag_reading, separator, lower)
    W2I = create_mapping(train, most_to_take=most_to_take)
    T2I = create_mapping(labels, ignore_elements={"Start-", "End-"})
    I2T = [key for key, value in sorted(T2I.iteritems(), key=lambda (k, v): (v, k))]
    train, dev = zip(train, labels), zip(dev, dev_labels)
    if '-s' in sys.argv:
        sub_map = sub_words_mapping(train, len(W2I))
        W2I.update(sub_map)
        my_network = MLPNetowrkSubWords

    vecs, W2I = prepare_pretrained(W2I)
    return train, dev, W2I, T2I, I2T, vecs, my_network


if __name__ == '__main__':
    train_pos, dev_pos, W2I_pos, T2I_pos, I2T_pos, vecs_pos, my_network = prepare_data(sys.argv[1], sys.argv[2], " ",
                                                                           most_to_take=40002)
    train_ner, dev_ner, W2I_ner, T2I_ner, I2T_ner, vecs_ner, my_network = prepare_data(sys.argv[3], sys.argv[4], "\t",
                                                                           most_to_take=20002)

    dev_pos_losses, dev_pos_accs = train_on_dataset(train_pos, dev_pos, 64, W2I_pos, T2I_pos, I2T_pos, 50,
                                                    "pos.model", dropout1=0.4, dropout2=0.4, emdM=vecs_pos,
                                                    network_constructor=my_network)
    dev_ner_losses, dev_ner_accs = train_on_dataset(train_ner, dev_ner, 64, W2I_ner, T2I_ner, I2T_ner,
                                                    50, "ner.model", dropout1=0.4, dropout2=0.4, emdM=vecs_ner,
                                                    network_constructor=my_network)
    plot_results(dev_pos_losses, "Model POS Loss", "Loss")
    plot_results(dev_pos_accs, "Model POS Accuracy", "Accuracy")

    plot_results(dev_ner_losses, "Model NER Loss", "Loss")
    plot_results(dev_ner_accs, "Model NER Accuracy", "Accuracy")
