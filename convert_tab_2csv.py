import os
from EnsembleDataset import EnsembleDataset


def loadfiles():
    with open("./Data/drugbank.tab", 'r') as f:
        lines = f.readlines()
    head = 'ID1,ID2,label,smile1,smile2'
    values = []
    for i in range(1, len(lines)):
        vs = lines[i].strip().split('\t')
        if len(vs) != 6:
            continue
        del vs[3]
        values.append(','.join([v for v in vs]))
    with open(f"Data/drugbank_all.csv", 'w') as f:
        f.writelines(head + '\n')
        for v in values:
            f.writelines(v + '\n')

    xlens = list(range(len(values)))
    nseg = 5
    segs = []
    for i in range(nseg):
        segs.append(xlens[i::nseg])
    for i in range(nseg):
        with open(f"Data/drugbank_train_{i}.csv", 'w') as f:
            f.writelines(head + '\n')
            for j in range(nseg):
                if i == j:
                    continue
                for l in segs[j]:
                    f.writelines(values[l] + '\n')
        with open(f"Data/drugbank_val_{i}.csv", 'w') as f:
            f.writelines(head + '\n')
            for l in segs[i]:
                f.writelines(values[l] + '\n')


def preprocessDict(train_set='./Data/drugbank_all.csv'):
    process_dataset = EnsembleDataset(train_set)


loadfiles()
preprocessDict()
