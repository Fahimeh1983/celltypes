import numpy as np

def cross_entropy(softmax_out, Y):
    """
    softmax_out: output out of softmax. shape: (vocab_size, m)
    """
    m = softmax_out.shape[1]
    cost = -(1 / m) * np.sum(np.sum(Y * np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
    return cost

# def cross_entropy(softmax_out, Y):
#     """
#     softmax_out: output out of softmax. shape: (vocab_size, m)
#     """
#     m = softmax_out.shape[1]
#     cost = -(1 / m) * np.sum(np.log(softmax_out[Y.flatten(), np.arange(Y.shape[1])] + 0.001))
#     return cost