import numpy as np
import os
from config import Config
import collections


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
PAD = "$PAD$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """
    Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```
    """
    def __init__(self, filename, 
                processing_word=None,
                processing_tag=None, 
                processing_tag_a=None,
                processing_tag_b=None,
                max_iter=None,
                use_char=False):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.processing_tag_a = processing_tag_a
        self.processing_tag_b = processing_tag_b
        self.max_iter = max_iter
        self.length = None
        self.use_char = use_char


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags, iob_tags, types, mentions, word_features = [], [], [], [], [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        mentions = find_mentions(iob_tags)
                        word_features = gen_word_features(words, self.processing_word, self.use_char)
                        yield words, tags, iob_tags, types, mentions, word_features
                        words, tags, iob_tags, types, mentions, word_features = [], [], [], [], [], []
                else:
                    ls = line.split()
                    word, tag = ls[0],ls[-1]
                    if tag != 'O':
                        tag_a = tag[0]
                        tag_b = tag[2:]
                    else:
                        tag_a = tag
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    if self.processing_tag_a is not None:
                        tag_boudry = self.processing_tag_a(tag_a)
                        iob_tags += [tag_boudry]
                    if self.processing_tag_b is not None and tag_a == 'B':
                        tag_type = self.processing_tag_b(tag_b)
                        types += [tag_type]
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def gen_word_features(words, processing_word, use_char=False):
    if use_char:
        word_ids = [t[1] for t in words]
        pad = processing_word(PAD)[1]
    else:
        word_ids = words
        pad = processing_word(PAD)
    sent_word_features = []
    
    for i in range(len(words)):
        word_features = []
        if i-4 < 0:
            word_features.append(pad)
        else:
            word_features.append(word_ids[i-4])
        
        if i-3 < 0:
            word_features.append(pad)
        else:
            word_features.append(word_ids[i-3])

        if i-2 < 0:
            word_features.append(pad)
        else:
            word_features.append(word_ids[i-2])

        if i-1 < 0:
            word_features.append(pad)
        else:
            word_features.append(word_ids[i-1])

        word_features.append(word_ids[i])

        sent_len = len(words)

        if i+1 >= sent_len:
            word_features.append(pad)
        else:
            word_features.append(word_ids[i+1])
        
        if i+2 >= sent_len:
            word_features.append(pad)
        else:
            word_features.append(word_ids[i+2])

        if i+3 >= sent_len:
            word_features.append(pad)
        else:
            word_features.append(word_ids[i+3])

        sent_word_features.append(word_features)
    return sent_word_features

def find_labels(iob, mentions, vocab_tags, types_dic):        
    labels = []
    mid = -1
    for idx, lab in enumerate(iob):
        if lab == 0:
            tag = 'O'
            labels.append(vocab_tags[tag])
        elif lab == 1:
            mid += 1
            tag = 'B-'+types_dic[mentions[mid]]
            #print tag
            labels.append(vocab_tags[tag])
        else:
            # TODO: fix the combining step here
            if mid >= len(mentions):
                labels.append(vocab_tags['O'])
            else:
                tag = 'I-'+types_dic[mentions[mid]]
                labels.append(vocab_tags[tag])
    return labels

def find_mentions(tags):
    mentions = []
    entity = []
    for idx, t in enumerate(tags):
        if t == 'B' or t == 1:
            if len(entity) > 0:
                mentions.append(entity)
                entity = []
            entity.append(idx)
        elif t == 'I' or t == 2:
            entity.append(idx)
        elif len(entity) > 0:
            mentions.append(entity)
            entity = []
    if len(entity) > 0:
        mentions.append(entity)
    return mentions

def get_vocabs(datasets):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags, _, _, _, _ in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """
    vocab_char = set()
    for words, _, _, _, _, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print("Building vocab...")
    count = 0
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
            count += 1
    print("- done. {} tokens!".format(count))
    return vocab


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK]

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        length_word = [max(map(lambda x: len(x), seq)) for seq in sequences if len(seq)>0]
        if len(length_word) == 0:
            max_length_word = 0
        else:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences if len(seq)>0])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)


    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch, iob_batch, type_batch, mention_batch, wfeat_batch = [], [], [], [], [], []
    for (x, y, a, b, c, d) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch, iob_batch, type_batch, mention_batch, wfeat_batch
            x_batch, y_batch, iob_batch, type_batch, mention_batch, wfeat_batch = [], [], [], [], [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]
        iob_batch += [a]
        type_batch += [b]
        mention_batch += [c]
        wfeat_batch += [d]

    if len(x_batch) != 0:
        left = minibatch_size - len(x_batch)
        for i in range(left):
            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x_batch[0]]
            y_batch += [y_batch[0]]
            iob_batch += [iob_batch[0]]
            type_batch += [type_batch[0]]
            mention_batch += [mention_batch[0]]
            wfeat_batch += [wfeat_batch[0]]
        yield x_batch, y_batch, iob_batch, type_batch, mention_batch, wfeat_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def merge_labels(boundry, seg_labels, idx_to_tag):
    output = []
    idx = 0
    for lab in boundry:
        if lab == 1:
            output.append(idx_to_tag[seg_labels[idx]])
            idx += 1
    return output

