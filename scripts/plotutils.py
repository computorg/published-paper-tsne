#%%
import plotly.express as px
import joblib

mem = joblib.Memory(".joblib", verbose = 0)

def plot2d(X, y, manifold_method, indices=None):
    X_transformed = mem.cache(manifold_method.fit_transform)(X)
    if indices is None:
        X_transformed = X_transformed[indices]
    fig = px.scatter(x=X_transformed[:,0], y=X_transformed[:,1],
            color=y)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend= {'itemsizing': 'constant'},
                      plot_bgcolor = "rgba(0, 0, 0, 0)")
    return fig

def plot3d(X, y, manifold_method, indices=None): 
    X_transformed = mem.cache(manifold_method.fit_transform)(X)
    if indices is None:
        X_transformed = X_transformed[indices]
    fig = px.scatter_3d(x=X_transformed[:,0], 
                        y=X_transformed[:,1], 
                        z=X_transformed[:,2],
            color=y)
    fig.update_traces(marker=dict(size=2, opacity=0.5),
                      selector=dict(mode='markers'))
    fig.update_scenes(xaxis_visible=False, 
                      yaxis_visible=False,
                      zaxis_visible=False )
    fig.update_layout(legend= {'itemsizing': 'constant'})
    return fig
