import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MultilabelF1Score, MultilabelAveragePrecision

def get_metric(metric_name):

    if metric_name == "meanIoU":
        return meanIoU
    elif metric_name == "f1Score":
        return f1_score
    elif metric_name == "subset_accuracy" or metric_name == "cls_f1Score" or metric_name == "mAP":
        return ClsMetric(metric_name=metric_name)
    else:
        raise RuntimeError("There is no metric named {}".format(metric_name))

class ClsMetric(object):
    def __init__(self, metric_name):
        if metric_name == "subset_accuracy":
            self.metric = MulticlassAccuracy(num_classes=10, average='macro')
        elif metric_name == "cls_f1Score":
            self.metric = MultilabelF1Score(num_labels=10, average='macro')
        elif metric_name == "mAP":
            self.metric = MultilabelAveragePrecision(num_labels=10, average="macro")
    def __call__(self, logits, label):
        assert len(logits.shape) == 2 and len(label.shape) == 2 and logits.shape == label.shape, f"Logit Shape: {logits.shape}, Label Shape: {label.shape}"
        probs = torch.sigmoid(logits)

        score = self.metric(probs, label)
        return score



def meanIoU(y_pred, y_true):

    C = y_pred.size()[-3]

    arg_pred = torch.argmax(y_pred, dim=-3)

    y_pred = F.one_hot(arg_pred, num_classes=C).permute(0, 3, 1, 2).to(torch.int32)
    y_true = F.one_hot(y_true, num_classes=C).permute(0, 3, 1, 2).to(torch.int32)

    inter = y_true & y_pred
    union = y_true | y_pred
    mIoU = torch.sum(inter, dim=(-2, -1))/(torch.sum(union, dim=(-2, -1))+1e-8)


    return torch.mean(mIoU)

def f1_score(y_pred, y_true):

    C = y_pred.size()[-3]
    
    arg_pred = torch.argmax(y_pred, dim=-3)
    
    y_pred = F.one_hot(arg_pred, num_classes=C).permute(0, 3, 1, 2).to(torch.int32)
    y_true = F.one_hot(y_true, num_classes=C).permute(0, 3, 1, 2).to(torch.int32)
    
    TP_2 = (y_true & y_pred) * 2
    TP_2_FN_FP = y_true + y_pred

    f1 = torch.sum(TP_2, dim=(-2, -1)).to(torch.float32) / (torch.sum(TP_2_FN_FP, dim=(-2, -1)).to(torch.float32)+1e-8)

    return torch.mean(f1)