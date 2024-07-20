import torch
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch_geometric.utils import negative_sampling
from tqdm.notebook import tqdm


def train_dynamic_sampling(edge_index, ratio=0.8):
    '''
    training時，學習結構的邊與計算損失的邊每個epoch(batch)會重新sample一次
    '''
    # 每一個 epoch 重新抽一次
    num_edges = int(edge_index.shape[1] * ratio)
    # 選中於學習結構的邊
    selected_edges = random.sample(range(edge_index.shape[1]), num_edges)
    selected_edge_index = edge_index[:, selected_edges]
    # 計算loss的邊
    remaining_edges = list(set(range(edge_index.shape[1])) - set(selected_edges))
    remaining_edges_index = edge_index[:, remaining_edges]

    return selected_edges, selected_edge_index, remaining_edges_index



# Implement the train function
def train(train_loader, model, neg, criterion, optimizer, device):
    model.to(device)
    model.train()

    total_loss = 0
    total_samples = 0
    for batch in train_loader:
        batch = batch.to(device)
        # 每一個batch都會重新抽取學習結構的邊和計算loss的邊
        selected_edges, selected_edge_index, remaining_edge_index = train_dynamic_sampling(batch.edge_index)
        selected_edges = torch.tensor(selected_edges, dtype=torch.long).to(device) # 學習結構的邊"索引"
        selected_edge_index = selected_edge_index.to(device) # 學習結構的邊
        remaining_edge_index = remaining_edge_index.to(device) # 計算loss的邊
        optimizer.zero_grad()

        # 生成負樣本
        # 負邊索引
        neg_edge_index = negative_sampling(
                        edge_index= batch.edge_index, num_nodes=batch.x.size(0), #selected_edge_index??避免負樣本選中計算loss的邊, 確保真的是負樣本
                        num_neg_samples= int(remaining_edge_index.size(1)*neg), method='sparse').to(device)
        
        # 計算loss的邊索引(正邊+負邊)
        all_edge_index = torch.cat([remaining_edge_index, neg_edge_index], dim=-1)
        # 計算loss的邊標籤
        edge_label = torch.cat([torch.ones(remaining_edge_index.size(1)),
                                torch.zeros(neg_edge_index.size(1))], dim=0).to(device)


        embedding = model(batch.x, selected_edge_index)
        pred = model.get_prediction(embedding, all_edge_index).view(-1)

        # optimization
        loss = criterion(pred, edge_label.float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)
        total_samples += pred.size(0)

    return total_loss / total_samples
    

@torch.no_grad()
def val(val_loader, model, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in val_loader:
        batch = batch.to(device)
        
        embedding = model(batch.x, batch.edge_index)        
        out = model.get_prediction(embedding, batch.edge_label_index).view(-1).sigmoid()
        all_preds.append(out.cpu().numpy())
        all_labels.append(batch.edge_label.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    roc_auc = roc_auc_score(all_labels, all_preds) #

    # 選擇F1-score最大的threshold
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
    F1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    F1_scores = np.nan_to_num(F1_scores) # 移除可能有nan
    max_index = np.argmax(F1_scores)
    best_threshold = thresholds[max_index] 
    
    return best_threshold, roc_auc, all_preds, all_labels


@torch.no_grad()
def test(test_loader, model, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in test_loader:
        batch = batch.to(device)

        embedding = model(batch.x, batch.edge_index)
        out = model.get_prediction(embedding, batch.edge_label_index).view(-1).sigmoid()
        all_preds.append(out.cpu().numpy())
        all_labels.append(batch.edge_label.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    roc_auc = roc_auc_score(all_labels, all_preds)

    return roc_auc, all_preds, all_labels