import dynet as dy
import numpy as np


class MLPNetwork(object):
    def __init__(self, pc, dims, vocab_size):
        self._expressions = list()
        num_params = len(dims) - 1

        for i in range(0, num_params, 2):
            self._expressions.append(pc.add_parameters((dims[i + 1], dims[i])))
            self._expressions.append(pc.add_parameters(dims[i + 1]))
        self._lookup = pc.add_lookup_parameters((vocab_size, dims[0] / 5))

    def __call__(self, inputs, droput=0.0):
        params = list()
        for exp in self._expressions:
            params.append(dy.parameter(exp))
        lookup = self._lookup
        word_1_batch = dy.lookup_batch(lookup, [x[0] for x in inputs])
        word_2_batch = dy.lookup_batch(lookup, [x[1] for x in inputs])
        word_3_batch = dy.lookup_batch(lookup, [x[2] for x in inputs])
        word_4_batch = dy.lookup_batch(lookup, [x[3] for x in inputs])
        word_5_batch = dy.lookup_batch(lookup, [x[4] for x in inputs])
        x = dy.concatenate([word_1_batch, word_2_batch, word_3_batch, word_4_batch, word_5_batch])
        if droput != 0.0:
            x = dy.dropout_batch(x, droput)
        scores_sym = params[2] * dy.tanh(
            params[0] * x + params[1]) + params[3]
        return scores_sym

    def forward(self, inputs, expected_output, droput=0.0):
        out = self(inputs, droput)
        loss = dy.sum_batches(dy.pickneglogsoftmax_batch(out, expected_output))
        predictions_probs = out.npvalue()
        if len(predictions_probs.shape) == 2:
            predictions = np.argmax(predictions_probs.T, axis=1)
        else:
            predictions = np.argmax(predictions_probs)
        return loss, predictions