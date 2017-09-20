from utils import get_trimmed_glove_vectors, load_vocab, get_processing_word, reverse_dictionary
from conll_data import CoNLLDataset
from model_ff import NERModel
from config import Config
import time
import collections

def main(config):
    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags  = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)
    vocab_iob = {"O":0, "B":1, "I":2}
    vocab_type = load_vocab(config.types_filename)
    id2type = reverse_dictionary(vocab_type)
    print vocab_type
    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars,
                    lowercase=True, chars=config.chars)
    processing_tag  = get_processing_word(vocab_tags, 
                    lowercase=False)
    processing_iob = get_processing_word(vocab_iob, 
                    lowercase=False)
    processing_type = get_processing_word(vocab_type, 
                    lowercase=False)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # create dataset
    dev   = CoNLLDataset(config.dev_filename, processing_word,
                        processing_tag, processing_iob, processing_type, config.max_iter, config.chars)
    test  = CoNLLDataset(config.test_filename, processing_word,
                        processing_tag, processing_iob, processing_type, config.max_iter, config.chars)
    train = CoNLLDataset(config.train_filename, processing_word,
                        processing_tag, processing_iob, processing_type, config.max_iter, config.chars)


    model = NERModel(config, embeddings, ntags=len(vocab_tags),
                                         nchars=len(vocab_chars),
                                         niob=len(vocab_iob),
                                         ntype=len(vocab_type),
                                         id2type=id2type)

    model.build()

    model.train(train, dev, vocab_tags)

    model.evaluate(test, vocab_tags)

if __name__ == "__main__":
    # create instance of config
    config = Config()
    
    # load, train, evaluate and interact with model
    main(config)
