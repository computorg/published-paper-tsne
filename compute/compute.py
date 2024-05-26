# %%
from sklearn import manifold
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_olivetti_faces, load_digits
# %%


def save_results(datasetname, X, y, n_components=2, methods={}):
    print(f"Computing for {datasetname}")
    multicol = pd.MultiIndex.from_product([methods.keys(), range(n_components)])
    pandas_df = pd.DataFrame(columns=multicol).sort_index(axis=1)
    for method in methods:
        print(f"Applying {method}")
        X_transformed = methods[method].fit_transform(X)
        for i in range(n_components):
            pandas_df[(method, i)] = X_transformed[:, i]

    pandas_df[("y", 0)] = y
    pandas_df.to_csv(f"{datasetname}.csv", index=False)


# %%

tsne = manifold.TSNE(
  n_components=2,
  perplexity=40,
  init="pca",
  random_state=42,
  verbose=True
)

sammon = manifold.MDS(
  n_components=2,
  normalized_stress='auto',
  n_jobs=-1,
  verbose=True
)

isomap = manifold.Isomap(
  n_components=2,
  n_neighbors=10,
  n_jobs=-1,
)

lle = manifold.LocallyLinearEmbedding(
  n_components=2,
  n_neighbors=10,
  n_jobs=-1,
)

methods = {
    "tsne": tsne,
    "sammon": sammon,
    "isomap": isomap,
    "lle": lle
}

# %%

data = load_digits()
X = data["data"]  # type: ignore
y = data["target"]  # type: ignore

save_results("digits", X, y, methods=methods)

data = fetch_olivetti_faces()
X = data["data"]  # type: ignore
y = data["target"]  # type: ignore

save_results("olivetti", X, y, methods=methods)

# %%
# reproduce the fig for random walk from the original tsne paper

data = fetch_openml("mnist_784", parser="auto")
X = data["data"]
y = data["target"]

single_tsne = manifold.TSNE(
  n_components=3,
  verbose=True
)

save_results("mnist", X, y, n_components=3, methods={"tsne": single_tsne})
# %%
