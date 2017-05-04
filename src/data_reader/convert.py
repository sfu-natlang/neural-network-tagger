#!/usr/bin/env python
#-*- coding: utf8 -*-


import sys
import time
import os

if __name__ == '__main__':


	f2 = open('/home/xkou/Downloads/en-ud-test.conllu','w')
	with open("/cs/natlang-data/CoNLL/universal-dependencies-1.2/UD_English/en-ud-test.conllu") as f:
		for l in f:
			line = l.strip()
			if not line :
				f2.write("\n")
				continue
			if line == '\n':
				f2.write("\n")
			else:
				tokens = line.split('\t')
				tokens[3] = tokens[4] # UPOS -> XPOS
				tokens = '\t'.join(tokens)
                      		f2.write(tokens+"\n")
	print "Done !!!"
