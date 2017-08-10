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

<h2>BiLSTM-CRF Model</h2>

Reference: https://github.com/guillaumegenthial/sequence_tagging

<h2>Mention2Vec with FF/BiLSTMs</h2>

python build_data.py
python main_ff.py/main.py

<h2>Experiments</h2>

Model                                                                                                           | POS  | NER | Chunk
--------------------------------------------------------------------------------------------------------------- | :---: | :---: | :-------:
SyntaxNet Greedy Model                                             | 97.20 |  - | -
GLM | 97.18 | 85.38 | 93.56
FeedForword (word) | 95.89 | - | -
FeedForword Model (history features) | 97.31 | -| -
Bi-LSTM (word) | 95.88 | 78.66 | -
Bi-LSTM (Character Embedding) | 97.08 | 78.87 | -
Bi-LSTM-CRF (Character Embedding) | 97.34 | - | -


<h2> Decoding Time (words/sec, with GPU enabled) </h2>

Model                                                                                                           | POS  | NER | Chunk
--------------------------------------------------------------------------------------------------------------- | :---: | :---: | :-------:
FeedForword Model (spelling features) | ~8000/s(~400/s) | - | -
Bi-LSTM-CRF (word feature only) | ~2000/s | - | -
Bi-LSTM-CRF (Character Embedding) | ~1500/s | - | -

 
