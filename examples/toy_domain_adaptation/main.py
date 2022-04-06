import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# from kale.embed.decomposition import MIDA
from kale.pipeline.multi_domain_adapter import _CoIRLS

# from adapt.feature_based import DANN
np.random.seed(81192)

N = 1000

Xs, ys = make_blobs(N, centers=[[0, 0], [0, 2]], cluster_std=[0.3, 0.35])
Xt, yt = make_blobs(N, centers=[[2, -2], [2, 0.2]], cluster_std=[0.35, 0.4])

plt.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap="coolwarm", alpha=0.4)
plt.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap="cool", alpha=0.4)
plt.title("Source domain and target domain blobs data", fontsize=14, fontweight="bold")

clf = RidgeClassifier(alpha=1.0)
clf.fit(Xs, ys)

ys_score = clf.decision_function(Xs)
yt_score = clf.decision_function(Xt)
yt_pred = clf.predict(Xt)
print("Accuracy on target domain: {:.2f}".format(accuracy_score(yt, yt_pred)))

covariates = np.zeros(N * 2)
covariates[:N] = 1
enc = OneHotEncoder(handle_unknown="ignore")
covariates_mat = enc.fit_transform(covariates.reshape(-1, 1)).toarray()

clf_ = _CoIRLS()
x = np.concatenate((Xs, Xt))
clf_.fit(x, ys, covariates_mat)

ys_score_ = clf_.decision_function(Xs)
yt_score_ = clf_.decision_function(Xt)
yt_pred_ = clf_.predict(Xt)
print("Accuracy on target domain: {:.2f}".format(accuracy_score(yt, yt_pred_)))

# coef_ = np.dot(x.T, da_learner.U)
