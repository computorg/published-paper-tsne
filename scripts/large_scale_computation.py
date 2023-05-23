###############################################################################
# Figure 2. Visualization by t-SNE and Sammon mapping
###############################################################################

from sklearn.datasets import load_digits
import numpy as np
from sklearn import manifold
import joblib

import matplotlib.pyplot as plt

mem = joblib.Memory(".joblib", verbose = 0)

digits = load_digits()

X = digits["data"]
y = digits["target"]

n_components = 2


colors = {
    0: "C0",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
    5: "C5",
    6: "C6",
    7: "C7",
    8: "C8",
    9: "C9",
}

fig, axes = plt.subplots(nrows=2, figsize=(6, 12))

###############################################################################
# t-SNE manifold learning
ax = axes[0]

tsne = manifold.TSNE(n_components=n_components, init="pca", random_state=42)
X_transformed = mem.cache(tsne.fit_transform)(X)
for label in np.unique(y):
    mask = y == label
    ax.scatter(
        X_transformed[mask, 0],
        X_transformed[mask, 1],
        c=colors[label], marker=".",
        label=label)

ax.spines["left"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("t-SNE")

###############################################################################
# Sammon mapping

# FIXME I think Sammon mapping has weights?
ax = axes[1]
embedding = manifold.MDS(n_components=n_components)
X_transformed = mem.cache(embedding.fit_transform)(X)
for label in np.unique(y):
    mask = y == label
    ax.scatter(
        X_transformed[mask, 0],
        X_transformed[mask, 1],
        c=colors[label], marker=".",
        label=label)

ax.spines["left"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Sammon mapping")
fig.savefig("sammon_mapping.png")

###############################################################################
# Figure 3. Visualization by Isomap and LLE
###############################################################################

fig, axes = plt.subplots(nrows=2, figsize=(6, 12))

###############################################################################
# ISOMAP
ax = axes[0]
isomap = manifold.Isomap(n_components=n_components, n_neighbors=10)
X_transformed = mem.cache(isomap.fit_transform)(X)
for label in np.unique(y):
    mask = y == label
    ax.scatter(
        X_transformed[mask, 0],
        X_transformed[mask, 1],
        c=colors[label], marker=".",
        label=label)

ax.spines["left"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Isomap")

###############################################################################
# LLE
ax = axes[1]
embedding = manifold.LocallyLinearEmbedding(
    n_components=n_components,
    n_neighbors=10)
X_transformed = mem.cache(embedding.fit_transform)(X)
for label in np.unique(y):
    mask = y == label
    ax.scatter(
        X_transformed[mask, 0],
        X_transformed[mask, 1],
        c=colors[label], marker=".",
        label=label)

ax.spines["left"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("LLE")

fig.savefig("isomap.png")
