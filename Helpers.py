import dynet as dy
import numpy as np
import matplotlib.pyplot as plt

def test_data(data, network, W2I, T2I):
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    for words, labels in data:
        dy.renew_cg()
        num_words = len(words) - 2
        words_nums = [tuple([W2I[word] if word in W2I else W2I["unknown"] for word in words[i - 2:i + 3]]) for i in
                      range(2, num_words)]
        labels_num = np.array([T2I[label] for label in labels[2:-2]])
        loss, prediction = network.forward(words_nums, labels_num)
        correct += np.sum(prediction == labels_num)
        total += prediction.size
        total_loss += loss.value()
    acc = correct / total
    total_loss /= total

    return total_loss, acc


def train_model(train, dev, network, trainer, W2I, T2I, num_iterations):
    dev_losses = list()
    dev_accs = list()
    for I in xrange(num_iterations):
        total_loss = 0.0
        correct = 0.0
        total = 0.0
        for words, labels in train:
            dy.renew_cg()
            num_words = len(words) - 2
            # Convert
            words_nums = [tuple([W2I[word] if word in W2I else W2I["unknown"] for word in words[i - 2:i + 3]]) for i in
                          range(2, num_words)]
            labels_num = np.array([T2I[label] for label in labels[2:-2]])
            loss, prediction = network.forward(words_nums, labels_num, 0.05)
            correct += np.sum(prediction == labels_num)
            total += prediction.size
            total_loss += loss.value()
            loss.backward()
            trainer.update()

        dev_loss, dev_acc = test_data(dev, network, W2I, T2I)
        dev_losses.append(dev_loss)
        dev_accs.append(dev_acc)

        print "Itertation:", I
        print "Training Loss:", total_loss / total
        print "Training Accuracy:", correct / total
        print "Dev Loss:", dev_loss
        print "Dev Accuracy:", dev_acc

    return dev_losses, dev_accs

# Plots the result of the training.
def plot_results(history, title, ylabel, xlabel='Epoch'):
    plt.plot(history)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

