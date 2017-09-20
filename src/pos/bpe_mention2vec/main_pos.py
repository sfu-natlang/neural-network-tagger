from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset
from model_pos import POSmodel
from config import Config
import time
import collections

def main(config):
    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags  = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)
    dictionary = load_vocab("data/types.txt")
    types_dic = collections.OrderedDict([(v, k) for k, v in dictionary.items()])
    vocab_iob = {"O":0, "B":1, "I":2}
    vocab_type = load_vocab(config.types_filename)
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

    ntype = len(vocab_type)
    model = POSmodel(config, embeddings, ntags=len(vocab_tags),
                                         nchars=len(vocab_chars),
                                         niob=3,
                                         ntype=ntype)

    model.build()

    model.train(train, dev, vocab_type)

    model.evaluate(test, vocab_type)

if __name__ == "__main__":
    # create instance of config
    config = Config()
    
    # load, train, evaluate and interact with model
    main(config)
