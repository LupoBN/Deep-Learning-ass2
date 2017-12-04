from Utils import *
from Helpers import *
from MLPNetwork import MLPNetwork
import sys

EMBEDDING_SIZE = 50


def train_on_dataset(train, dev, hidden_layer, learning_rate, W2I, T2I, I2T, num_of_iterations, save_file, droput1=0.0,
                     droput2=0.0, emdM=None, ner=False, network_constructor=MLPNetwork):
    m = dy.ParameterCollection()
    # create network
    network = network_constructor(m, [EMBEDDING_SIZE * 5, hidden_layer, hidden_layer, len(T2I)], len(W2I), W2I, T2I,
                                  I2T, emdM)
    # create trainer
    trainer = dy.AdamTrainer(m)
    dev_losses, dev_accs = train_model(train, dev, network, trainer,
                                       num_of_iterations, save_file, droput1, droput2, ner)
    return dev_losses, dev_accs


def test_model(dev, model_file, hidden_layer, W2I, T2I, I2T):
    m = dy.ParameterCollection()

    # create network
    network = MLPNetwork(m, [EMBEDDING_SIZE * 5, hidden_layer, hidden_layer, len(T2I)], len(W2I), W2I, T2I, I2T)

    network.load_model(model_file)

    dev_loss, dev_acc = test_data(dev, network)
    print dev_loss, dev_acc


def prepare_data(train_data_file, dev_data_file, separator, lower=False, most_to_take=15000):
    train, labels = read_file(train_data_file, parse_tag_reading, separator, lower)
    dev, dev_labels = read_file(dev_data_file, parse_tag_reading, separator, lower)
    W2I = create_mapping(train, most_to_take=most_to_take)
    T2I = create_mapping(labels, ignore_elements={"Start-", "End-"})
    I2T = [key for key, value in sorted(T2I.iteritems(), key=lambda (k, v): (v, k))]
    return zip(train, labels), zip(dev, dev_labels), W2I, T2I, I2T


if __name__ == '__main__':
    train_pos, dev_pos, W2I_pos, T2I_pos, I2T_pos = prepare_data("data/pos/train", "data/pos/dev", " ",
                                                                 most_to_take=12000)
    train_ner, dev_ner, W2I_ner, T2I_ner, I2T_ner = prepare_data("data/ner/train", "data/ner/dev", "\t",
                                                                 most_to_take=20000)

    dev_pos_losses, dev_pos_accs = train_on_dataset(train_pos, dev_pos, 64, 0.01, W2I_pos, T2I_pos, I2T_pos, 50,
                                                    "encoding_pos.model")
    dev_ner_losses, dev_ner_accs = train_on_dataset(train_ner, dev_ner, 64, 0.01, W2I_ner, T2I_ner, I2T_ner,
                                                    50, "encoding_ner.model", droput1=0.4, droput2=0.4)
    plot_results(dev_pos_losses, "Model POS Loss", "Loss")
    plot_results(dev_pos_accs, "Model POS Accuracy", "Accuracy")

    plot_results(dev_ner_losses, "Model NER Loss", "Loss")
    plot_results(dev_ner_accs, "Model NER Accuracy", "Accuracy")
