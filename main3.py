## Stage3: Link Prediction
### Utilized this newly synthesized dataset for link prediction, aiming to reconstruct the original structure of the social network.
import torch
import pickle
from torch_geometric.loader import DataLoader
from stage3.model import MyGNN
import argparse
from stage3.training_utils import train, val, test
from stage3.eval_plot import evaluate_threshold, plot_metrics, evaluate_predictions


def main(args):
    use_data = args.use_data
    use_model = args.use_model
    lr = args.lr
    n_epoch = args.n_epoch
    trytry = args.trytry

    neg = 1  # 訓練時生成幾倍負樣本
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 匯入資料
    with open(f'dataset/{use_data}/train100.pickle', 'rb') as file:
        dataset_train = pickle.load(file)

    with open(f'dataset/{use_data}/valid20.pickle', 'rb') as file:
        dataset_valid = pickle.load(file)

    with open(f'dataset/{use_data}/test20.pickle', 'rb') as file:
        dataset_test = pickle.load(file)


    # 建立 DataLoader
    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset_valid, batch_size=8)
    test_loader = DataLoader(dataset_test, batch_size=8)

    # Start Link prediction with GNN
    model = MyGNN(conv_type=use_model).to(device)
    print(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss_record = []
    threshold_record = []
    val_auc_record = []
    val_preds_record = []
    val_labels_record = []
    test_auc_record = []
    test_preds_record = []
    test_labels_record = []

    for epoch in range(n_epoch):
        loss = train(train_loader, model, neg, criterion, optimizer, device)
        threshold, val_auc, val_preds, val_labels = val(val_loader, model, device)
        test_auc, test_preds, test_labels = test(test_loader, model, device)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, ValAUC: {val_auc:.4f}, TestAUC: {test_auc:.4f}, threshold: {threshold:.4f}')

        loss_record.append(loss)
        threshold_record.append(threshold)
        val_auc_record.append(val_auc)
        val_preds_record.append(val_preds)
        val_labels_record.append(val_labels)
        test_auc_record.append(test_auc)
        test_preds_record.append(test_preds)
        test_labels_record.append(test_labels)

    # 儲存模型
    torch.save(model, f'model/{use_data}/2_{use_model}_model_{trytry}.pt')

    # 根據 threshold(0.5) 計算 acc, pr, re, f1, cm
    tr_acc, tr_pr, tr_re, tr_f1, tr_cm = evaluate_threshold(test_preds_record[-1], test_labels_record[-1], 0.5)
    general_result = {"acc": tr_acc, "pr": tr_pr, "re": tr_re, "f1": tr_f1, "cm": tr_cm}
    print(f"Testing Result when threshold = 0.5: {general_result}")

    # 使用最後一個epoch的threshold紀錄
    use_threshold = threshold_record[-1]

    val_acc_record, val_pr_record, val_re_record, val_f1_record, val_cm_record = evaluate_predictions(use_threshold, val_preds_record, val_labels_record)
    test_acc_record, test_pr_record, test_re_record, test_f1_record, test_cm_record = evaluate_predictions(use_threshold, test_preds_record, test_labels_record)

    evaluation_record = {
        "loss": loss_record,
        "val_auc": val_auc_record, "test_auc": test_auc_record,
        "test_acc": test_acc_record, "test_pr": test_pr_record, "test_re": test_re_record, "test_f1": test_f1_record, "test_cm": test_cm_record,
        "val_acc": val_acc_record, "val_pr": val_pr_record, "val_re": val_re_record, "val_f1": val_f1_record, "val_cm": val_cm_record,
        "threshold": threshold_record
    }
    with open(f'result/{use_data}/2_{use_model}_Evaluation_{trytry}.pickle', 'wb') as file:
        pickle.dump(evaluation_record, file)
    print(f"已將 evaluation_record 存至 result/{use_data}/2_{use_model}_Evaluation_{trytry}.pickle")

    final_result = {
        "loss": loss_record[-1], "AUC": test_auc_record[-1],
        "ACC": test_acc_record[-1], "PR": test_pr_record[-1], "RE": test_re_record[-1], "F1": test_f1_record[-1], "CM": test_cm_record[-1], 
        "threshold": threshold_record[-1]
    }
    print(f"Final Testing Result: {final_result}")

    plot_metrics(use_data, use_model, trytry, n_epoch, loss_record, val_auc_record, test_auc_record, val_acc_record, test_acc_record, val_pr_record, test_pr_record, val_re_record, test_re_record, val_f1_record, test_f1_record, threshold_record)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNNExaplainer attack') 
    parser.add_argument('--use_data', type=str, default='Facebook', help='Enter the dataset name (Facebook, Github)')
    parser.add_argument('--use_model', type=str, default="GCN", help='Enter the model to use (GCN, GAT, GraphSAGE)')
    parser.add_argument('--lr', type=float, default=0.01, help='Enter the learning rate (e.g., 0.005)')
    parser.add_argument('--n_epoch', type=int, default=1, help='Enter the number of epochs (e.g., 150)')
    parser.add_argument('--trytry', type=int, default=999, help='Enter the trial number (e.g., 999)')
    args = parser.parse_args()
    main(args)
