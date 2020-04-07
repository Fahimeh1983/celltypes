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


def get_idx2word(vocabulary):
    '''
    take the vocabulary and generate the index to word and word to index

    Parameters
    ----------
    vocabulary: a set of all the unique words

    Returns
    -------
    a dictionary with the indices as keys and words as values
    '''
    return {idx: w for (idx, w) in enumerate(vocabulary)}


def get_word2idx(vocabulary):
    '''
    take the vocabulary and generate the index to word and word to index

    Parameters
    ----------
    vocabulary: a set of all the unique words

    Returns
    -------
    a dictionary with the words as keys and indices as values
    '''
    return {w: idx for (idx, w) in enumerate(vocabulary)}


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
            last_context_word_index = min(i + window, len(text))
            for j in range(first_context_word_index, last_context_word_index):
                if i != j:
                    context_tuple_list.append((word, text[j]))
    print("There are {} pairs of target and context words".format(len(context_tuple_list)))
    return context_tuple_list