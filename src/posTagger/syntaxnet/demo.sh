#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# A script that runs a tokenizer, a part-of-speech tagger and a dependency
# parser on an English text file, with one sentence per line.
#
# Example usage:
#  echo "Parsey McParseface is my favorite parser!" | syntaxnet/demo.sh

# To run on a conll formatted file, add the --conll command line argument.

bazel-bin/syntaxnet/parser_trainer \
	--task_context=syntaxnet/context.pbtxt \
	--arg_prefix=brain_pos \
	--compute_lexicon \
	--graph_builder=greedy \
	--training_corpus=training-corpus \
	--tuning_corpus=tuning-corpus \
	--output_path=models \
	--batch_size=32 \
	--decay_steps=3600 \
	--hidden_layer_sizes=128 \
	--learning_rate=0.08 \
	--momentum=0.9 \
	--seed=1 \
	--params=128-0.08-3600-0.9-0 
