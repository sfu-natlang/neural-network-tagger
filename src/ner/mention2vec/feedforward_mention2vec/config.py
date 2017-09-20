import os
from general_utils import get_logger


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # create instance of logger
        self.logger = get_logger(self.log_path)
        

    # general config
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    # embeddings
    dim = 100
    dim_char = 50
    glove_filename = "/cs/natlang-user/vivian/NeuroNER/data/word_vectors/glove.6B.100d.txt"
    
    # trimmed embeddings (created from glove_filename with build_data.py)
    trimmed_filename = "data/glove.6B.{}d.trimmed.npz".format(dim)

    # dataset
    dev_filename = "/cs/natlang-user/vivian/eng.testa"
    test_filename = "/cs/natlang-user/vivian/eng.testb"
    train_filename = "/cs/natlang-user/vivian/eng.train"
    max_iter = None # if not None, max number of examples

    # vocab (created from dataset with build_data.py)
    words_filename = "data/words.txt"
    tags_filename = "data/tags.txt"
    chars_filename = "data/chars.txt"
    types_filename= "data/types.txt"
    
    # training
    train_embeddings = True
    nepochs = 50
    dropout = 0.5
    batch_size = 32
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    clip = -1 # if negative, no clipping
    nepoch_no_imprv = 10
    reload = False
    
    # model hyperparameters
    hidden_size = 100
    char_hidden_size = 50
    
    # NOTE: if both chars and crf, only 1.6x slower on GPU
    crf = True# if crf, training is 1.7x slower on CPU
    chars = True # if char embedding, training is 3.5x slower on CPU

