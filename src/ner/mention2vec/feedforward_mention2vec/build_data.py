from config import Config
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from utils import get_vocabs, get_glove_vocab, write_vocab, load_vocab, \
    get_char_vocab, export_trimmed_glove_vectors, get_processing_word
from conll_data import CoNLLDataset, UNK, NUM, PAD


def build_data(config):
    """
    Procedure to build data

    Args:
        config: defines attributes needed in the function
    Returns:
        creates vocab files from the datasets
        creates a npz embedding file from trimmed glove vectors
    """
    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = CoNLLDataset(config.dev_filename, processing_word)
    test  = CoNLLDataset(config.test_filename, processing_word)
    train = CoNLLDataset(config.train_filename, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.glove_filename)
    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)
    vocab = list(vocab)
    vocab.insert(0, PAD)

    # Save vocab
    write_vocab(vocab, config.words_filename)
    write_vocab(vocab_tags, config.tags_filename)

    # Trim GloVe Vectors
    vocab = load_vocab(config.words_filename)
    export_trimmed_glove_vectors(vocab, config.glove_filename, 
                                config.trimmed_filename, config.dim)

    # Build and save char vocab
    train = CoNLLDataset(config.train_filename, processing_word)
    vocab_chars = get_char_vocab(train)
    vocab_chars = list(vocab_chars)
    vocab_chars.insert(0, PAD)
    write_vocab(vocab_chars, config.chars_filename)


    # Build and save type vocab
    vocab_types = set()
    print len(vocab_tags)
    for tag in vocab_tags:
        if tag != 'O':
            vocab_types.add(tag[2:])
    write_vocab(vocab_types, config.types_filename)


if __name__ == "__main__":
    config = Config()
    build_data(config)
