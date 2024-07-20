import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 根據 threshold 計算 acc, pr, re, f1, cm
def evaluate_threshold(preds, labels, threshold):
    preds = preds > threshold
    acc = accuracy_score(labels, preds)
    pr = precision_score(labels, preds)
    re = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return acc, pr, re, f1, cm


# 給定最後一個epoch的threshold & 每個樣本的pred label, 計算 acc, pr, re, f1, cm
def evaluate_predictions(threshold, preds_record, labels_record):
    # Initialize records for evaluation metrics
    acc_record = []
    pr_record = []
    re_record = []
    f1_record = []
    cm_record = []

    # Iterate over each epoch's results
    for preds, labels in zip(preds_record, labels_record):
        preds_thresholded = preds > threshold

        acc = accuracy_score(labels, preds_thresholded)
        precision = precision_score(labels, preds_thresholded)
        recall = recall_score(labels, preds_thresholded)
        f1 = f1_score(labels, preds_thresholded)
        cm = confusion_matrix(labels, preds_thresholded)
        
        acc_record.append(acc)
        pr_record.append(precision)
        re_record.append(recall)
        f1_record.append(f1)
        cm_record.append(cm)
        
    return acc_record, pr_record, re_record, f1_record, cm_record



def plot_metrics(data_name, model_name, try_name, n_epoch, loss_record, val_auc_record, test_auc_record, val_acc_record, test_acc_record, val_pr_record, test_pr_record, val_re_record, test_re_record, val_f1_record, test_f1_record, threshold_record):
    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.2)

    # Plot loss
    plt.subplot(1,3,1)
    plt.plot(range(n_epoch), loss_record)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    # Plot AUC
    plt.subplot(1,3,2)
    plt.plot(range(n_epoch), val_auc_record, label='valid')
    plt.plot(range(n_epoch), test_auc_record, label='test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Valid/Testing AUC")

    # Plot Accuracy
    plt.subplot(1,3,3)
    plt.plot(range(n_epoch), val_acc_record, label='valid')
    plt.plot(range(n_epoch), test_acc_record, label='test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title("Valid/Testing Accuracy")

    # Save the figure
    plt.savefig(f'plot/{data_name}/2_{model_name}_lossAUC_{try_name}.png', dpi=300)
    print(f"已儲存圖片至 plot/{data_name}/2_{model_name}_lossAUC_{try_name}.png")
    plt.close()

    # Plot Precision, Recall, and F1-score
    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.2)
    plt.subplot(1, 3, 1)
    plt.plot(range(n_epoch), val_pr_record, label='valid')
    plt.plot(range(n_epoch), test_pr_record, label='test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Valid/Testing Precision")

    plt.subplot(1, 3, 2)
    plt.plot(range(n_epoch), val_re_record, label='valid')
    plt.plot(range(n_epoch), test_re_record, label='test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Valid/Testing Recall")

    plt.subplot(1 ,3, 3)
    plt.plot(range(n_epoch), val_f1_record, label='valid')
    plt.plot(range(n_epoch), test_f1_record, label='test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Valid/Testing F1-score")

    plt.savefig(f'plot/{data_name}/2_{model_name}_PrReF1_{try_name}.png', dpi=300)
    print(f"已儲存圖片至 plot/{data_name}/2_{model_name}_PrReF1_{try_name}.png")
    plt.close()

    # Plot Threshold
    plt.figure(figsize=(6, 4)) 
    plt.plot(range(n_epoch), threshold_record)
    plt.xlabel("Epoch")
    plt.ylabel("Threshold")
    plt.title("Best threshold for each epoch during Validation")
    plt.savefig(f'plot/{data_name}/2_{model_name}_threshold_{try_name}.png', dpi=300)
    print(f"已儲存圖片至 plot/{data_name}/2_{model_name}_threshold_{try_name}.png")
    plt.close()

