from Utils import *
from Helpers import *
from MLPNetwork import MLPNetwork

EMBEDDING_SIZE = 50
HIDDEN_DIM = 30


def train_on_dataset(train_file, dev_file, seperator, num_of_iterations, save_file, droput1=0.0, droput2=0.0):
    m = dy.ParameterCollection()
    train, labels = read_file(train_file, parse_tag_reading, seperator)
    dev, dev_labels = read_file(dev_file, parse_tag_reading, seperator)
    W2I = create_mapping(train, 5)
    T2I = create_mapping(labels, ignore_elements={"Start-", "End-"})
    # create network
    network = MLPNetwork(m, [EMBEDDING_SIZE * 5, HIDDEN_DIM, HIDDEN_DIM, len(T2I)], len(W2I))
    # create trainer
    trainer = dy.SimpleSGDTrainer(m, learning_rate=0.01)
    dev_losses, dev_accs = train_model(zip(train, labels), zip(dev, dev_labels), network, trainer, W2I,
                                       T2I, num_of_iterations, save_file, droput1, droput2)
    return dev_losses, dev_accs

def test_model(train_file, dev_file, seperator, model_file):
    m = dy.ParameterCollection()
    train, labels = read_file(train_file, parse_tag_reading, seperator)
    dev, dev_labels = read_file(dev_file, parse_tag_reading, seperator)
    W2I = create_mapping(train, 5)
    T2I = create_mapping(labels, ignore_elements={"Start-", "End-"})
    # create network
    network = MLPNetwork(m, [EMBEDDING_SIZE * 5, HIDDEN_DIM, HIDDEN_DIM, len(T2I)], len(W2I))
    network.load_model(model_file)

    dev_loss, dev_acc = test_data(zip(dev, dev_labels), network, W2I, T2I)
    print dev_loss, dev_acc


if __name__ == '__main__':


    dev_pos_losses, dev_pos_accs = train_on_dataset("data/pos/train", "data/pos/dev", " ", 50, "encoding_pos.model",
                                                    0.6, 0.6)
    dev_ner_losses, dev_ner_accs = train_on_dataset("data/ner/train", "data/ner/dev", "\t", 50, "encoding_ner.model")

    plot_results(dev_pos_losses, "Model POS Loss", "Loss")
    plot_results(dev_pos_accs, "Model POS Accuracy", "Accuracy")

    plot_results(dev_ner_losses, "Model NER Loss", "Loss")
    plot_results(dev_ner_accs, "Model NER Accuracy", "Accuracy")
    test_model("data/pos/train", "data/pos/dev", " ", "encoding_pos.model")
    test_model("data/ner/train", "data/ner/dev", "\t", "encoding_ner.model")