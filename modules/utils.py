import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt



class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

        self._dims=(0,-1,-2)
        self.eps = 0.01

    def forward(self, out_prob, gt_one_hot):
        intersection = torch.sum(out_prob*gt_one_hot,dim=self._dims)
        dice = (2 * intersection + self.eps)/ (torch.sum(gt_one_hot,dim=self._dims)+ torch.sum(out_prob,dim=self._dims) + self.eps)
        return torch.mean(1. - dice[1:])


def calc_dice_score(cm_matrix):
    dices = np.zeros((cm_matrix.shape[0]))

    tp = np.diagonal(cm_matrix)
    sumPred = cm_matrix.sum(axis=0)
    sumGt = cm_matrix.sum(axis=1)
    valid = sumGt > 0

    dices[valid] = np.divide(2*tp[valid], sumGt[valid] + sumPred[valid])
    return dices


def calc_iou_scores(cm_matrix):
    true_positives = np.diag(cm_matrix)
    false_positives = np.sum(cm_matrix, axis=0) - true_positives
    false_negatives = np.sum(cm_matrix, axis=1) - true_positives

    iou_scores = true_positives / (true_positives + false_positives + false_negatives + 1e-10)

    return iou_scores


def plot_confusion_matrix_asym(cm, class_names_x, class_names_y, title="Confusion matrix"):
    plt.ioff()
    figure = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    #plt.colorbar()
    tick_marks_x = np.arange(len(class_names_x))
    plt.xticks(tick_marks_x, class_names_x, rotation=45)
    tick_marks_y = np.arange(len(class_names_y))
    plt.yticks(tick_marks_y, class_names_y)
    
    # Normalize the confusion matrix.
    cm = np.around(cm, decimals=3)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_values(dice_list, class_names, title="Dices"):
    dices = np.around(np.array(dice_list).reshape((1,len(dice_list))), decimals=2)
    values = dices

    plt.ioff()
    figure = plt.figure(figsize=(6, 2.75))
    plt.imshow(values, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(np.arange(1))

    threshold = values.max() / 2.
    for i, j in itertools.product(range(values.shape[0]), range(values.shape[1])):
        color = "white" if values[i, j] > threshold else "black"
        plt.text(j, i, values[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    return figure