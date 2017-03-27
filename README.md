 # neural-network-tagger
A General Purpose Tagger for POS Tagging, NER Tagging, and Chunking.

Part-OF-Speech Tagging

1. Prepare WSJ Data
a. convert to conll format 
https://gist.github.com/khlmnn/3cc07407a002bb1773cd
b. map XPOSTAG to UPOSTAG before training using convert.py

2. Install Syntaxnet 
https://github.com/tensorflow/models/tree/a9133ae914b44602c5f26afbbd7dd794ff9c6637/syntaxnet
3. Train and test the model using taggerTrain.sh, taggerTest.sh and tagger.pbtxt

Model                                                                                                           | News  | Training Time | Test Time
--------------------------------------------------------------------------------------------------------------- | :---: | :---: | :-------:
SyntaxNet Greedy Model (12 CPUs)                                              | 97.20 |  1424.66s | 3.02s
GLM | 97.18 | - | -
NN Tagger | 97.31 | - | -
NN Tagger with BPE | - | - | -
 
