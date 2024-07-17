# %%
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.io as pio
import os

pio.templates.default = "plotly_white"
if os.environ["QUARTO_FIG_FORMAT"] == "pdf":
    pio.renderers.default = "pdf"

def plot2d(X, y, manifold_method):
    fig = px.scatter(
        x=X[:, 0],  # type: ignore
        y=X[:, 1],  # type: ignore
        color=y,
        category_orders={"color": y.cat.categories},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend={"itemsizing": "constant"})
    return fig


def plot3d(X, y, manifold_method):
    fig = px.scatter_3d(
        x=X[:, 0],  # type: ignore
        y=X[:, 1],  # type: ignore
        z=X[:, 2],  # type: ignore
        color=y,  # type: ignore
        category_orders={"color": y.cat.categories},
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
    y = pandas_df["y"].iloc[:, 0].astype("category") # type: ignore
    X = pandas_df.drop(columns=["y"])

    for method in methods:
        if X[method].shape[1] == 2:
            plot2d(X[method].values, y, method).show()
        elif X[method].shape[1] == 3:
            plot3d(X[method].values, y, method).show()
