from __future__ import print_function
import numpy as np
import torch
from sklearn.metrics import f1_score

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def splits(dataset_str, data, num_classes, num_nodes, deg):

    for _ in range(50):
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

            train_index = torch.cat([i[:20] for i in indices], dim=0)
            test_index = torch.cat([i[20:] for i in indices], dim=0)

        train_mask = index_to_mask(train_index, size=data.num_nodes)
        test_mask = index_to_mask(test_index, size=data.num_nodes)
        write_mode = 'a' if _ != 0 else 'w'
        with open(dataset_str + '_split2_train_50.txt', write_mode) as f:
            f.write(','.join(map(str, torch.arange(0, num_nodes)[train_mask].tolist())) + '\n')
        with open(dataset_str + '_split2_test_50.txt', write_mode) as f:
            f.write(','.join(map(str, torch.arange(0, num_nodes)[test_mask].tolist())) + '\n')
    return data



def classify(dataset_str, embeds, labels, eval_iter):
    # run = wandb.init()
    from sklearn.linear_model import LogisticRegression

    import torch
    num_nodes = embeds.shape[0]


    acc_all = []

    with open(dataset_str + "_split_train_50.txt", 'r') as f:
        train_idx_list = f.readlines()
    with open(dataset_str + "_split_test_50.txt", 'r') as f:
        test_idx_list = f.readlines()

    for _ in range(eval_iter):

        train_mask = torch.zeros(num_nodes).bool()
        train_mask[[int(i) for i in train_idx_list[_].split(',')]] = True
        test_mask = torch.zeros(num_nodes).bool()
        test_mask[[int(i) for i in test_idx_list[_].split(',')]] = True
        clf = LogisticRegression(solver='liblinear', max_iter=400).fit(embeds[train_mask], labels[train_mask])
        output = torch.LongTensor(clf.predict(embeds[test_mask]))
        acc = (output == labels[test_mask]).sum() / (test_mask).sum()
        acc_all.append(acc)

    print('acc:', np.mean(acc_all)*100, np.std(acc_all)*100)





