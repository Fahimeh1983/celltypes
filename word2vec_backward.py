import numpy as np

def softmax_backward(Y, softmax_out):
    """
    Y: labels of training data. shape: (vocab_size, m)
    softmax_out: output out of softmax. shape: (vocab_size, m)
    """
    dL_dZ = softmax_out - Y

    assert (dL_dZ.shape == softmax_out.shape)
    return dL_dZ


# def softmax_backward(Y, softmax_out):
#     """
#     Y: labels of training data. shape: (vocab_size, m)
#     softmax_out: output out of softmax. shape: (vocab_size, m)
#     """
#     m = Y.shape[1]
#     softmax_out[Y.flatten(), np.arange(m)] -= 1.0
#     dL_dZ = softmax_out
#
#     assert (dL_dZ.shape == softmax_out.shape)
#     return dL_dZ

def dense_backward(dL_dZ, caches):
    """
    dL_dZ: shape: (vocab_size, m)
    caches: dict. results from each steps of forward propagation
    """
    W = caches['W']
    word_vec = caches['word_vec']
    m = word_vec.shape[1]

    dL_dW = (1 / m) * np.dot(dL_dZ, word_vec.T)
    dL_dword_vec = np.dot(W.T, dL_dZ)

    assert (W.shape == dL_dW.shape)
    assert (word_vec.shape == dL_dword_vec.shape)

    return dL_dW, dL_dword_vec


def backward_propagation(Y, softmax_out, caches):
    dL_dZ = softmax_backward(Y, softmax_out)
    dL_dW, dL_dword_vec = dense_backward(dL_dZ, caches)

    gradients = dict()
    gradients['dL_dZ'] = dL_dZ
    gradients['dL_dW'] = dL_dW
    gradients['dL_dword_vec'] = dL_dword_vec

    return gradients


def update_parameters(parameters, caches, gradients, learning_rate):
    vocab_size, emb_size = parameters['WRD_EMB'].shape
    inds = caches['inds']
    WRD_EMB = parameters['WRD_EMB']
    dL_dword_vec = gradients['dL_dword_vec']
    m = inds.shape[-1]

    WRD_EMB[inds.flatten(), :] -= dL_dword_vec.T * learning_rate

    parameters['W'] -= learning_rate * gradients['dL_dW']