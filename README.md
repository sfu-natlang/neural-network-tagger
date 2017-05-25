 # neural-network-tagger
A General Purpose Tagger for POS Tagging, NER Tagging, and Chunking.


Syntaxnet

1. Prepare WSJ Data for Part-OF-Speech Tagging
a. convert to conll format 
https://gist.github.com/khlmnn/3cc07407a002bb1773cd
b. map XPOSTAG to UPOSTAG before training using convert.py

2. Install Syntaxnet 
https://github.com/tensorflow/models/tree/a9133ae914b44602c5f26afbbd7dd794ff9c6637/syntaxnet
3. Train and test the model using taggerTrain.sh, taggerTest.sh and tagger.pbtxt

FeedForward Model

cd PATH_TO_TAGGER/src/feedforward_model
Training: python tagger_trainer.py 
Evaluating: python tagger_eval.py

Bi-LSTM-CRF Model
cd PATH_TO_TAGGER/src/lstm_crf_model
POS Training: python lstm_char_train.py
NER Training: python ner_train.py
NER Eval: perl conlleval.pl < ner_out

Result

Model                                                                                                           | POS  | NER | Chunk
--------------------------------------------------------------------------------------------------------------- | :---: | :---: | :-------:
SyntaxNet Greedy Model (12 CPUs)                                              | 97.20 (97.44 reported) |  - | -
GLM | 97.18 | 85.38 | -
FeedForword Model (spelling features) | 97.31 | - | -
FeedForword (BPE) | 96.64 | - | -
Bi-LSTM-CRF (word feature only) | 95.88 | 82.00 | -
Bi-LSTM-CRF (Character Embedding) | 97.37(*) | - | -
Bi_LSTM-CRF (spelling feature) | 96.98 | - | -

 
