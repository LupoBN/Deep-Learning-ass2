from Utils import *
from Helpers import *
from MLPNetwork import MLPNetwork
EMBEDDING_SIZE = 50
HIDDEN_DIM = 30

if __name__ == '__main__':
    m = dy.ParameterCollection()
    train_pos, labels_pos = read_file("data/pos/train", parse_tag_reading, " ")
    W2I = create_mapping(train_pos, 5)
    T2I = create_mapping(labels_pos, ignore_elements={"Start-", "End-"})
    # create network
    network = MLPNetwork(m, [EMBEDDING_SIZE * 5, HIDDEN_DIM, HIDDEN_DIM, len(T2I)], len(W2I))
    # create trainer
    trainer = dy.SimpleSGDTrainer(m, learning_rate=0.01)
    dev_pos, dev_labels_pos = read_file("data/pos/dev", parse_tag_reading, " ")

    dev_losses, dev_accs = train_model(zip(train_pos, labels_pos), zip(dev_pos, dev_labels_pos), network, trainer, W2I,
                                       T2I, 100)
