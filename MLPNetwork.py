import dynet as dy
import numpy as np


class MLPNetwork(object):
    def __init__(self, pc, dims, vocab_size, W2I, T2I, I2T, pre_lookup=None):
        self._expressions = list()
        num_params = len(dims) - 1

        for i in range(0, num_params, 2):
            self._expressions.append(pc.add_parameters((dims[i + 1], dims[i])))
            self._expressions.append(pc.add_parameters(dims[i + 1]))
        if pre_lookup is None:
            self._lookup = pc.add_lookup_parameters((vocab_size, dims[0] / 5))
        else:
            self._lookup = pc.lookup_parameters_from_numpy(pre_lookup)

        self._pc = pc
        self._W2I = W2I
        self._T2I = T2I
        self._I2T = I2T

    def _create_word_batch(self, inputs, i):
        return dy.lookup_batch(self._lookup, [x[i] for x in inputs])

    def _words_operation(self, inputs):
        num_inputs = [tuple([self._W2I[word] if word in self._W2I else self._W2I["UUUNKKK"] for word in quid])
                      for quid in inputs]
        word_batches = [self._create_word_batch(num_inputs, i) for i in range(0, 5)]
        return dy.concatenate(word_batches)

    def __call__(self, inputs, droput1=0.0, droput2=0.0):
        params = list()
        for exp in self._expressions:
            params.append(dy.parameter(exp))
        x = self._words_operation(inputs)
        if droput1 != 0.0:
            x = dy.dropout_batch(x, droput1)
        second_layer_activation = dy.tanh(params[0] * x + params[1])
        if droput2 != 0.0:
            second_layer_activation = dy.dropout(second_layer_activation, droput2)
        scores_sym = params[2] * second_layer_activation + params[3]
        return scores_sym

    def forward(self, inputs, expected_output, droput1=0.0, droput2=0.0):
        out = self(inputs, droput1, droput2)
        expected_output = np.array([self._T2I[exp] for exp in expected_output])
        loss = dy.sum_batches(dy.pickneglogsoftmax_batch(out, expected_output))
        predictions_probs = out.npvalue()

        if len(predictions_probs.shape) == 2:
            predictions = np.argmax(predictions_probs.T, axis=1)
        else:
            predictions = np.array([np.argmax(predictions_probs)])
        predictions = np.array([self._I2T[prediction] for prediction in predictions])
        return loss, predictions

    def predict(self, inputs):
        out = self(inputs)
        predictions_probs = out.npvalue()

        if len(predictions_probs.shape) == 2:
            predictions = np.argmax(predictions_probs.T, axis=1)
        else:
            predictions = np.array([np.argmax(predictions_probs)])
        predictions = np.array([self._I2T[prediction] for prediction in predictions])
        return predictions



    def save_model(self, file_name):
        self._pc.save(file_name)

    def load_model(self, file_name):
        self._pc.populate(file_name)


class MLPNetowrkSubWords(MLPNetwork):
    def __init__(self, pc, dims, vocab_size, W2I, T2I, I2T, pre_lookup=None):
        super(MLPNetowrkSubWords, self).__init__(pc, dims, vocab_size, W2I, T2I, I2T, pre_lookup)

    def _words_operation(self, inputs):
        num_inputs = [tuple([self._W2I[word] if word in self._W2I else self._W2I["UUUNKKK"]
                             for word in quid]) for quid in inputs]
        num_prefixes = [
            tuple([self._W2I["Pre-" + word[0:3]] if "Pre-" + word[0:3] in self._W2I else self._W2I["Pre-UNK"]
                   for word in quid]) for quid in inputs]

        num_suffixes = [tuple([self._W2I["Suf-" + word[-4:-1]] if "Suf-" + word[-4:-1] in self._W2I
                               else self._W2I["Suf-UNK"] for word in quid]) for quid in inputs]
        word_batches = [self._create_word_batch(num_inputs, i) for i in range(0, 5)]
        prefix_batch = [self._create_word_batch(num_prefixes, i) for i in range(0, 5)]
        suffix_batch = [self._create_word_batch(num_suffixes, i) for i in range(0, 5)]
        sum_batches = [word_batches[i] + prefix_batch[i] + suffix_batch[i] for i in range(0, 5)]
        return dy.concatenate(sum_batches)
