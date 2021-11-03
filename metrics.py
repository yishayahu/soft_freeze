import numpy as np
from sklearn.metrics import roc_auc_score
def computeAUROC(dataGT, dataPRED, nnClassCount):
    # Computes area under ROC curve
    # dataGT: ground truth data
    # dataPRED: predicted data
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(nnClassCount):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            pass
    return np.array(outAUROC).mean()