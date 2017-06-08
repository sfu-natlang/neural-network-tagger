 # neural-network-tagger
A General Purpose Tagger for POS Tagging, NER Tagging, and Chunking.


<h2>Syntaxnet</h2>

1. Prepare WSJ Data for Part-OF-Speech Tagging
a. convert to conll format 
https://gist.github.com/khlmnn/3cc07407a002bb1773cd
b. map XPOSTAG to UPOSTAG before training using convert.py

2. Install Syntaxnet 
https://github.com/tensorflow/models/tree/a9133ae914b44602c5f26afbbd7dd794ff9c6637/syntaxnet
3. Train and test the model using taggerTrain.sh, taggerTest.sh and tagger.pbtxt

<h2>FeedForward Model</h2>

cd PATH_TO_TAGGER/src/feedforward_model

Training: python tagger_trainer.py 

Evaluating: python tagger_eval.py

<h2>Bi-LSTM-CRF Model</h2>

cd PATH_TO_TAGGER/src/lstm_crf_model

POS Training: python lstm_char_train.py

NER Training: python ner_train.py

NER Eval: perl conlleval.pl < ner_out

<h2>Experiments</h2>

Model                                                                                                           | POS  | NER | Chunk
--------------------------------------------------------------------------------------------------------------- | :---: | :---: | :-------:
SyntaxNet Greedy Model                                             | 97.20 |  - | -
GLM | 97.18 | 85.38 | -
FeedForword Model (spelling features) | 97.31 | - | -
FeedForword (BPE) | 96.64 | - | -
Bi-LSTM-CRF (word feature only) | 95.88 | - | -
Bi-LSTM-CRF (Character Embedding) | 97.08 | 78.87 | -


<h2> Time (words/sec, with GPU enabled) </h2>

Model                                                                                                           | POS  | NER | Chunk
--------------------------------------------------------------------------------------------------------------- | :---: | :---: | :-------:
GLM | ~800/s | - | -
FeedForword Model (spelling features) | ~8000/s (1)  | - | -
FeedForword (BPE) | ~2500/s (1) | - | -
Bi-LSTM-CRF (word feature only) | - | - | -
Bi-LSTM-CRF (Character Embedding) | ~1500/s (2) | - | -

(1) not including the time for precomputing the features for feedforword model

(2) online training
 
