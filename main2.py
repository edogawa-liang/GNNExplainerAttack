import os
import pickle
import numpy as np
import os
import pickle
import argparse
import torch
from stage2 import gen, syn

'''
# 將所有的dataset都加一個idx屬性 (還沒做)
for i in range(len(dataset_train)):
    dataset_train[i]['idx'] = i
for i in range(len(dataset_valid)):
    dataset_valid[i]['idx'] = i
for i in range(len(dataset_test)):
    dataset_test[i]['idx'] = i
'''

os.environ['TORCH'] = torch.__version__
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Stage2:
    def __init__(self, usedata, use_existing, dtype, num, ratio, epoch):
        self.usedata = usedata
        self.use_existing = use_existing
        self.num = num
        self.dtype = dtype
        self.ratio = ratio
        self.epoch = epoch

    def get_data(self):
        exp_dir = f'explanations/{self.usedata}'
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
        filepath = f'{exp_dir}/exp_{self.dtype}{self.num}_{self.epoch}.pickle'

        GSubgs = gen.GeneratingSubgraphs(self.usedata, 3, self.epoch)
        SynD = syn.SynData(self.dtype, self.ratio)

        self.dataset_list = []
        if self.use_existing:
            print("Subgraphs information loaded from file.")
            with open(filepath, 'rb') as file:
                explanations = pickle.load(file)

            for i in range(self.num):
                subgraphs = GSubgs.generate_or_load_subgraphs(self.use_existing, explanations[i])
                dataset = SynD.process(subgraphs)
                self.dataset_list.append(dataset)
        else:
            print("GNNExplainer is generating subgraphs...")
            explainers_list = []
            for i in range(self.num):
                explainers, subgraphs = GSubgs.generate_or_load_subgraphs(self.use_existing)
                explainers_list.append(explainers)
                dataset = SynD.process(subgraphs)
                self.dataset_list.append(dataset)

            with open(filepath, 'wb') as file:
                pickle.dump(explainers_list, file)

        return self.dataset_list

    def cal_stat(self):
        ns, es, ls = [], [], []
        for dataset in self.dataset_list:
            ns.append(dataset.x.shape[0])
            es.append(dataset.edge_index.shape[1])
            ls.append(dataset.edge_label.shape[0])

        avg_nodes = np.mean(ns)
        if self.dtype == "train":
            avg_edges = np.mean(es) * self.ratio
            avg_labels = np.mean(ls) * (1 - self.ratio) * 2
        else:
            avg_edges = np.mean(es)
            avg_labels = np.mean(ls)

        stat = {
            "avg_nodes": avg_nodes,
            "avg_edges": avg_edges,
            "avg_labels": avg_labels
        }
        print(f"Average number of nodes: {int(round(avg_nodes, 0))}")
        print(f"Average number of given edges: {int(round(avg_edges, 0))}")
        print(f"Average number of edges to predict (including negative edge): {int(round(avg_labels, 0))}")
        return stat


def main(args):
    stage2 = Stage2(args.usedata, args.use_existing, args.dtype, args.num, args.ratio, args.epoch)
    dataset_list = stage2.get_data()
    stat = stage2.cal_stat()

    dataset_dir = f'dataset/{args.usedata}'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    with open(f'{dataset_dir}/{args.dtype}{args.num}_epoch{args.epoch}.pickle', 'wb') as file:
        pickle.dump(dataset_list, file)
    with open(f'{dataset_dir}/stat_{args.dtype}{args.num}_epoch{args.epoch}.pickle', 'wb') as file:
        pickle.dump(stat, file)

    print(f"Dataset for {args.usedata} {args.dtype}{args.num}_epoch{args.epoch} saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating Explanatory Subgraphs and New Dataset Synthesis")
    parser.add_argument('--usedata', type=str, default='Facebook', help='Enter the dataset name (Facebook, Github)')
    parser.add_argument('--use_existing', type=bool, default=False, help='Use existing subgraphs? (True/False)')
    parser.add_argument('--dtype', type=str, default='train', help='Enter the data type (e.g., train, valid, test)')
    parser.add_argument('--num', type=int, default=1, help='Enter the number of subgraphs to generate')
    parser.add_argument('--ratio', type=float, default=0.8, help='Ratio for training data splitting')
    parser.add_argument('--epoch', type=int, default=1, help='Enter the number of epochs for GNNExplainer')

    args = parser.parse_args()
    main(args)
