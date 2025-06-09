import torch
import torch.nn as nn
import torch.nn.functional as F

def get_metric(metric_name):

    if metric_name == "meanIoU":
        return meanIoU
    elif metric_name == "f1Score":
        return f1_score
    else:
        raise RuntimeError("There is no metric named {}".format(metric_name))

def meanIoU(y_pred, y_true):

    C = y_true.size()[-3]

    S1 = y_pred[:,[a for a in range(C-1)],:,:]
    S1 = torch.argmax(S1, dim=-3)
    S1 = F.one_hot(S1, num_classes=C-1).permute(0, 3, 1, 2).to(torch.int32)

    latSacrum = y_pred[:,[a for a in range(C-2)] + [C-1],:,:]
    latSacrum = torch.argmax(latSacrum, dim=-3)
    latSacrum = F.one_hot(latSacrum, num_classes=C-1).permute(0, 3, 1, 2).to(torch.int32)
    
    y_pred = torch.concat([S1[:,:-1,:,:]&latSacrum[:,:-1,:,:], S1[:,-1,:,:].unsqueeze(1), latSacrum[:,-1,:,:].unsqueeze(1)],dim=-3)


    # 자 이후 처리를 생각해보자 | or & ,etc.
    y_pred = y_pred.to(torch.int32)
    y_true = y_true.to(torch.int32)

    inter = y_true & y_pred
    union = y_true | y_pred
    mIoU = torch.sum(inter, dim=(-2, -1))/(torch.sum(union, dim=(-2, -1))+1e-8)


    return torch.mean(mIoU)

def f1_score(y_pred, y_true):

    C = y_true.size()[-3]
    S1 = y_pred[:,[a for a in range(C-1)],:,:]
    S1 = torch.argmax(S1, dim=-3)
    S1 = F.one_hot(S1, num_classes=C-1).permute(0, 3, 1, 2).to(torch.int32)

    latSacrum = y_pred[:,[a for a in range(C-2)] + [C-1],:,:]
    latSacrum = torch.argmax(latSacrum, dim=-3)
    latSacrum = F.one_hot(latSacrum, num_classes=C-1).permute(0, 3, 1, 2).to(torch.int32)

    y_pred = torch.concat([S1[:,:-1,:,:]&latSacrum[:,:-1,:,:], S1[:,-1,:,:].unsqueeze(1), latSacrum[:,-1,:,:].unsqueeze(1)],dim=-3)

    # 자 이후 처리를 생각해보자 | or & ,etc.
    y_pred = y_pred.to(torch.int32)
    y_true = y_true.to(torch.int32)
    
    TP_2 = (y_true & y_pred) * 2
    TP_2_FN_FP = y_true + y_pred

    f1 = torch.sum(TP_2, dim=(-2, -1)).to(torch.float32) / (torch.sum(TP_2_FN_FP, dim=(-2, -1)).to(torch.float32)+1e-8)

    return torch.mean(f1)