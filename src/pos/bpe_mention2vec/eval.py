import numpy as np
file1 = "output"
file2 = "/cs/natlang-user/vivian/wsj-conll/test.conllu"
tags1 = []
with open(file1) as f:
    for line in f:
        line = line.strip()
        if (len(line) == 0 or line.startswith("-DOCSTART-")):
            continue
        else:
            ls = line.split()
            tag = ls[-1]
            tags1 += [tag]
tags2 = []
with open(file2) as f:
    for line in f:
        line = line.strip()
        if (len(line) == 0 or line.startswith("-DOCSTART-")):
            continue
        else:
            ls = line.split()
            tag = ls[4]
            tags2 += [tag]
print tags1
print tags2
accs = [a==b for (a, b) in zip(tags2, tags1)]
acc = np.mean(accs)
print acc
