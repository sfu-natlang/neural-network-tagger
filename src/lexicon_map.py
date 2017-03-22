import pickle
from data.data_pool import DataPool

prefix = []
suffix = []
prefix3 = []
suffix3 = []
def buildLexiconTable(sent):
  for word in sent:
    if len(word) > 1 and word[0:2] not in prefix:
      prefix.append(word[0:2])
    if len(word) > 2 and word[0:3] not in prefix3:
      prefix3.append(word[0:3])
    if len(word) > 1 and word[-2:] not in suffix:
      suffix.append(word[-2:])
    if len(word) > 2 and word[-3:] not in suffix3:
      suffix3.append(word[-3:])

def writeListToFile():
  with open('prefix-list', 'wb') as fp:
    pickle.dump(prefix, fp)
  with open('suffix-list', 'wb') as fp:
    pickle.dump(suffix, fp) 
  with open('prefix-list3', 'wb') as fp:
    pickle.dump(prefix3, fp)
  with open('suffix-list3', 'wb') as fp:
    pickle.dump(suffix3, fp)

def test():
  with open('prefix-list', 'rb') as fp:
    itemlist = pickle.load(fp)
  for word in itemlist:
    print "%s" %word
  print ""
  with open('suffix-list', 'rb') as fb:
    itemlist = pickle.load(fb)
  for word in itemlist:
    print "%s" %word

def readInSent():
  config = {}
  config['format'] = 'penn2malt'
  config['train'] = 'wsj_0[2-9][0-9][0-9].mrg.3.pa.gs.tab|wsj_1[0-9][0-9][0-9].mrg.3.pa.gs.tab|wsj_2[0-1][0-9][0-9].mrg.3.pa.gs.tab'
  config['data_path'] = '/cs/natlang-user/vivian/penn-wsj-deps'
  trainDataPool = DataPool(data_format  = config['format'],
                           data_regex   = config['train'],
                           data_path    = config['data_path'])
  sents = []
  while trainDataPool.has_next_data():
    sents.append(trainDataPool.get_next_data().get_word_list())
  return sents

if __name__ == "__main__":
  sents = readInSent()
  for sent in sents:
    buildLexiconTable(sent)
  writeListToFile()
  print len(prefix)
  print len(suffix)
  print len(prefix3)
  print prefix3
  print len(suffix3)

