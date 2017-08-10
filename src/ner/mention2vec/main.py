from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset
from model import NERModel
from config import Config
import time

def main(config):
    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags  = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)
    vocab_iob = {"O":0, "B":1, "I":2}
    vocab_type = {"LOC":0, "PER":1, "ORG":2, "MISC":3}

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
                        processing_tag, processing_iob, processing_type, config.max_iter)
    test  = CoNLLDataset(config.test_filename, processing_word,
                        processing_tag, processing_iob, processing_type, config.max_iter)
    train = CoNLLDataset(config.train_filename, processing_word,
                        processing_tag, processing_iob, processing_type, config.max_iter)


    model = NERModel(config, embeddings, ntags=len(vocab_tags),
                                         nchars=len(vocab_chars),
                                         niob=3,
                                         ntype=4)

    model.build()

    # train, evaluate and interact
    print vocab_tags
    model.train(train, dev, vocab_tags)
    stime = time.time()
    
    model.evaluate(test, vocab_tags)
    
    etime = time.time()
    print etime-stime

if __name__ == "__main__":
    # create instance of config
    config = Config()
    
    # load, train, evaluate and interact with model
    main(config)
