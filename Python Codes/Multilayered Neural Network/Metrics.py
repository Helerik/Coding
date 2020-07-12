#!/usr/bin/env python3
# Author: Erik Davino Vincent

import numpy as np

# Metrics class for storing different evaluation methods
class Metrics():

    @classmethod
    def __true_positives(cls, true_Y, predicted_Y, average):
        if average == "micro":
            return len(np.where((predicted_Y == true_Y) == True)[0])
        elif average == "macro":
            TPs = []
            for tag in range(np.max(true_Y)+1):
                   TPs.append(len(np.where((predicted_Y[np.where(predicted_Y == tag)] == true_Y[np.where(predicted_Y == tag)]) == True)[0]))
            return np.array(TPs)

    @classmethod
    def __false_positives(cls, true_Y, predicted_Y, average):
        if average == "micro":
            return len(np.where((predicted_Y == true_Y) == False)[0])
        elif average == "macro":
            FPs = []
            for tag in range(np.max(true_Y)+1):
                FPs.append(len(np.where((predicted_Y[np.where(predicted_Y == tag)] == true_Y[np.where(predicted_Y == tag)]) == False)[0]))
            return np.array(FPs)

    @classmethod
    def __false_negatives(cls, true_Y, predicted_Y, average):
        if average == "micro":
            return len(np.where((predicted_Y != true_Y) == True)[0])
        elif average == "macro":
            FNs = []
            for tag in range(np.max(true_Y)+1):
                FNs.append(len(np.where((predicted_Y[np.where(true_Y == tag)] != true_Y[np.where(true_Y == tag)]) == True)[0]))
            return np.array(FNs)

    @classmethod
    def accuracy(cls, true_Y, predicted_Y):
        percent = np.mean((predicted_Y == true_Y).astype(int))
        return percent

    @classmethod
    def precision(cls, true_Y, predicted_Y, average = "micro"):
        TP = cls.__true_positives(true_Y, predicted_Y, average)
        FP = cls.__false_positives(true_Y, predicted_Y, average)
        return np.divide(TP, (TP + FP))

    @classmethod
    def recall(cls, true_Y, predicted_Y, average = "micro"):
        TP = cls.__true_positives(true_Y, predicted_Y, average)
        FN = cls.__false_negatives(true_Y, predicted_Y, average)
        return np.divide(TP, (TP + FN))
    
    @classmethod
    def f1_score(cls, true_Y, predicted_Y, average = "micro"):
        if average == "micro":
            precision = cls.precision(true_Y, predicted_Y, average)
            recall = cls.recall(true_Y, predicted_Y, average)
            return 2*precision*recall/(precision + recall)
        elif average == "macro":
            precision = cls.precision(true_Y, predicted_Y, average)
            recall = cls.recall(true_Y, predicted_Y, average)
            return np.mean(2*precision*recall/(precision + recall))
        elif average == "_all":
            precision = cls.precision(true_Y, predicted_Y, "macro")
            recall = cls.recall(true_Y, predicted_Y, "macro")
            return 2*precision*recall/(precision + recall)

    @classmethod
    def score_table(cls, true_Y, predicted_Y):
        precision_macro = cls.precision(true_Y, predicted_Y, "macro")
        recall_macro = cls.recall(true_Y, predicted_Y, "macro")

        f1_micro = cls.f1_score(true_Y, predicted_Y, "micro")
        f1_macro = cls.f1_score(true_Y, predicted_Y, "macro")
        f1_all = cls.f1_score(true_Y, predicted_Y, "_all")
        
        accuracy = cls.accuracy(true_Y, predicted_Y)

        ret_str =  f"""         Score Table:

    |    Accuracy: {accuracy:.2%}
    |
    |    Tag     Recall     Precision     F1-Score"""
        for tag in range(0, len(f1_all)):
            ret_str += f"\n    |     {tag} :     {recall_macro[tag]:.2f}       {precision_macro[tag]:.2f}          {f1_all[tag]:.2f}"
        ret_str += "\n    |    " + f"\n    |    Micro F1-Score: {f1_micro:.2f}    Macro F1-Score: {f1_macro:.2f}"

        return ret_str
