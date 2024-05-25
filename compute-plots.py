# %%
from sklearn import manifold
import joblib
import plotly.graph_objects as go

mem = joblib.Memory(".joblib", bytes_limit='16G',verbose = 0)

n_components = 2
tsne = manifold.TSNE(
  n_components=n_components,
  perplexity=40, 
  init="pca",
  random_state=42,
  n_jobs=-1)

sammon = manifold.MDS(
  n_components=2,
  normalized_stress='auto',
  n_jobs=-1)

isomap = manifold.Isomap(
  n_components=n_components,
  n_neighbors=10,
  n_jobs=-1)

lle = manifold.LocallyLinearEmbedding(
  n_components=n_components,
  n_neighbors=10,
  n_jobs=-1)

methods = [tsne, sammon, isomap, lle]

def plot2d(X, y, plot_func, manifold_method, indices = None):
    X_transformed = mem.cache(manifold_method.fit_transform)(X)
    mem.reduce_size()
    if not (indices is None):
        X_transformed = X_transformed[indices]
        y = y[indices]
    fig = plot_func(X_transformed, y)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend= {'itemsizing': 'constant'},
                      plot_bgcolor = "rgba(0, 0, 0, 0)")
    fig.show()

# %%
#| label: fig-mnist
#| fig-cap: "Visualizations of 6,000 handwritten digits from the MNIST dataset."
#| fig-subcap: 
#|   - "Visualization by t-SNE."
#|   - "Visualization by Sammon mapping."
#|   - "Visualization by Isomap."
#|   - "Visualization by LLE."
#| layout-ncol: 2

import plotly.express as px
import numpy as np
from sklearn.datasets import fetch_openml

digits = fetch_openml("mnist_784", parser="auto")

X = digits["data"][:1000]
y = digits["target"][:1000]
sorted_idx = np.argsort(y)

def plot_mnist(X, y):
    fig = px.scatter(x=X[:,0], y=X[:,1], color=y)
    return fig

for method in methods:
   plot2d(X, y, plot_mnist, method, sorted_idx)

# %%
#| label: fig-olivetti
#| fig-cap: "Visualization of the Olivetti faces data set"
#| fig-subcap: 
#|   - "Visualization by t-SNE."
#|   - "Visualization by Sammon mapping."
#|   - "Visualization by Isomap."
#|   - "Visualization by LLE."
#| layout-ncol: 2
#| error: false

from sklearn.datasets import fetch_olivetti_faces
from plotly.validators.scatter.marker import SymbolValidator

data = fetch_olivetti_faces()
X = data["data"]
y = data["target"]

def plot_olivetti(X, y):
    fig = px.scatter(x=X[:,0], y=X[:,1],symbol=y.astype(str),color=y.astype(str))
    fig = fig.update_traces(marker=dict(size=8),
                            selector=dict(mode='markers'))
    fig = fig.update_layout(showlegend=False)
    return fig
# get the plot for each methods
for method in methods:
    plot2d(X, y, plot_olivetti, method)
# %%
np.unique(y, return_counts=True)

#%%
def plot2d(X, y, manifold_method, indices=None):
    X_transformed = mem.cache(manifold_method.fit_transform)(X)
    if indices is None:
        X_transformed = X_transformed[indices]
    fig = px.scatter(x=X_transformed[:,0], y=X_transformed[:,1],
            symbol=y,colors=colors, markers=markers)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend= {'itemsizing': 'constant'},
                      plot_bgcolor = "rgba(0, 0, 0, 0)")
    return fig

###############################################################################
# t-SNE manifold learning
fig, ax = plt.subplots()

tsne = manifold.TSNE(n_components=n_components, init="pca", random_state=42)
X_transformed = mem.cache(tsne.fit_transform)(X)
for label, (m, c) in zip(np.unique(y), itertools.product(markers, colors)):
    mask = y == label
    ax.scatter(
        X_transformed[mask, 0],
        X_transformed[mask, 1],
	marker=m, c=c,
	s=8,
        label=label)

ax.spines["left"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("t-SNE")
plt.show()
###############################################################################
# Sammon mapping

fig, ax = plt.subplots()
embedding = manifold.MDS(n_components=n_components)
X_transformed = mem.cache(embedding.fit_transform)(X)
for label, (m, c) in zip(np.unique(y), itertools.product(markers, colors)):
    mask = y == label
    ax.scatter(
        X_transformed[mask, 0],
        X_transformed[mask, 1],
	marker=m, c=c,
	s=8,
        label=label)

ax.spines["left"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Sammon mapping")

###############################################################################
# ISOMAP
fig, ax = plt.subplots()
isomap = manifold.Isomap(n_components=n_components, n_neighbors=10)
X_transformed = mem.cache(isomap.fit_transform)(X)
for label, (m, c) in zip(np.unique(y), itertools.product(markers, colors)):
    mask = y == label
    ax.scatter(
        X_transformed[mask, 0],
        X_transformed[mask, 1],
	marker=m, c=c,
	s=8,
        label=label)

ax.spines["left"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Isomap")
plt.show()
###############################################################################
# LLE
fig, ax = plt.subplots()
embedding = manifold.LocallyLinearEmbedding(
    n_components=n_components,
    n_neighbors=10)
X_transformed = mem.cache(embedding.fit_transform)(X)

for label, (m, c) in zip(np.unique(y), itertools.product(markers, colors)):
    mask = y == label
    ax.scatter(
        X_transformed[mask, 0],
        X_transformed[mask, 1],
	marker=m, c=c,
	s=8,
        label=label)

ax.spines["left"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("LLE")
plt.show()

# %%
