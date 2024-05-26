# %%
import plotly.express as px
import pandas as pd
import numpy as np


def plot2d(X, y, manifold_method, indices=None):
    if indices is None:
        X = X[indices]  # type: ignore
    fig = px.scatter(
        x=X[:, 0],  # type: ignore
        y=X[:, 1],  # type: ignore
        color=y
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        legend={"itemsizing": "constant"}, plot_bgcolor="rgba(0, 0, 0, 0)"
    )
    return fig


def plot3d(X, y, manifold_method, indices=None):
    if indices is None:
        X = X[indices]  # type: ignore
    fig = px.scatter_3d(
        x=X[:, 0],  # type: ignore
        y=X[:, 1],  # type: ignore
        z=X[:, 2],  # type: ignore
        color=y,  # type: ignore
    )
    fig.update_traces(
        marker=dict(size=2, opacity=0.5),
        selector=dict(mode="markers")
    )
    fig.update_scenes(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False
    )
    fig.update_layout(legend={"itemsizing": "constant"})
    return fig


methods = ["tsne", "sammon", "isomap", "lle"]


def plot_dataset(datasetname, methods=methods, sample_size=None):
    pandas_df = pd.read_csv(f"{datasetname}.csv", header=[0, 1])
    if sample_size is not None:
        pandas_df = pandas_df.sample(sample_size)
    y = pandas_df["y"].iloc[:, 0]  # type: ignore
    sorted_idx = np.argsort(y)
    X = pandas_df.drop(columns=["y"])

    for method in methods:
        if X[method].shape[1] == 2:
            plot2d(X[method].values, y, method, sorted_idx)
        elif X[method].shape[1] == 3:
            plot3d(X[method].values, y, method, sorted_idx)


