import itertools

def get_vocabulary(corpus):
    '''
    Get all the sentences as a list of lists and return the unique vocabulary

    Parameters
    ----------
    corpus: All the sentences as list of lists

    Returns
    -------
    vocabulary: a set of all the unique words in the corpus
    '''
    return set(itertools.chain.from_iterable(corpus))


def get_idx2word(vocabulary, padding=False):
    '''
    take the vocabulary and generate the index to word and word to index

    Parameters
    ----------
    vocabulary: a set of all the unique words
    padding: if true then index zero will be for padding

    Returns
    -------
    a dictionary with the indices as keys and words as values
    '''

    idx2word = {}
    if padding:
        print("a node called pad is added for padding and its index is zero")
        idx2word[0] = 'pad'
        for idx, w in enumerate(vocabulary):
            idx2word[idx+1] = w
    else:
        idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    return idx2word

def get_word2idx(vocabulary, padding=False):
    '''
    take the vocabulary and generate the index to word and word to index

    Parameters
    ----------
    vocabulary: a set of all the unique words
    padding: if true then index zero will be for padding

    Returns
    -------
    a dictionary with the words as keys and indices as values
    '''
    word2idx = {}
    if padding:
        word2idx['pad'] = 0
        print("a node called pad is added for padding and its index is zero")
        for idx, w in enumerate(vocabulary):
            word2idx[w] = idx + 1
    else:
        word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}

    return word2idx


def get_word_context_tuples(corpus, window):
    '''

    Parameters
    ----------
    corpus: list of list, which each list is tokenized sentence
    window: target-context window size

    Returns
    -------
    context_tuple_list: a list of tuples which the first member of the tuple is the word and the second
    member in the tuple is the context
    '''

    context_tuple_list = []

    for text in corpus:
        for i, word in enumerate(text):
            first_context_word_index = max(0, i - window)
            last_context_word_index = min(i + window + 1, len(text))
            for j in range(first_context_word_index, last_context_word_index):
                if i != j:
                    context_tuple_list.append((word, text[j]))
    print("There are {} pairs of target and context words".format(len(context_tuple_list)))
    return context_tuple_list

def MCBOW_get_word_context_tuples(corpus, window):
    '''

    Parameters
    ----------
    corpus: list of list, which each list is tokenized sentence
    window: target-context window size

    Returns
    -------
    context_tuple_list: a list of tuples which the first member of the tuple is the word and the second
    member in the tuple is the context
    '''
    print("MCBOW by default adds a padding node called pad with index zero")
    context_tuple_list = []

    for text in corpus:
        for i, word in enumerate(text):
            first_context_word_index = max(0, i - window)
            last_context_word_index = min(i + window + 1, len(text))

            context_list = [text[j] for j in range(first_context_word_index, last_context_word_index) if i != j]
            if len(context_list) < window * 2:
                pad_size = window * 2 - len(context_list)
                for i in range(pad_size):
                    context_list.append('pad')
            context_tuple_list.append((word, context_list))

    print("There are {} pairs of target and context words".format(len(context_tuple_list)))
    return context_tuple_list

