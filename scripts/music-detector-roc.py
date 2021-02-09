#!/usr/bin/env python3

import sys

import pandas
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt


data_3way = pandas.read_csv(sys.argv[1])
data_dnsmos = pandas.read_csv(sys.argv[2])

data_3way['label'] = data_3way.filename.apply(lambda s: int(s[0]))

(fpr_3way, tpr_3way, _) = sklearn.metrics.roc_curve(
    data_3way.label, data_3way.music)

(fpr_dnsmos, tpr_dnsmos, _) = sklearn.metrics.roc_curve(
    data_dnsmos.ground_truth, data_dnsmos.pred_probability)

auc_3way = sklearn.metrics.roc_auc_score(
    data_3way.label, data_3way.music)

auc_dnsmos = sklearn.metrics.roc_auc_score(
    data_dnsmos.ground_truth, data_dnsmos.pred_probability)

pandas.DataFrame({
    "fpr_3way": fpr_3way,
    "tpr_3way": tpr_3way
}).to_csv("music-detector-roc-MobileNetV1.csv")

pandas.DataFrame({
    "fpr_dnsmos": fpr_dnsmos,
    "tpr_dnsmos": tpr_dnsmos,
}).to_csv("music-detector-roc-DNSMOS.csv")

plt.figure(figsize=(4, 4))

plt.plot(fpr_3way, tpr_3way, label="MobileNetV1: AUC = %.4f" % auc_3way)
plt.plot(fpr_dnsmos, tpr_dnsmos, label="DNSMOS: AUC = %.4f" % auc_dnsmos)

plt.title("Music detector ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.tight_layout()

plt.savefig("music-detector-roc.png", dpi=200)
# plt.show()
