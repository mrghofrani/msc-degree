# %%
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.options.plotting.backend = "plotly"

# %%

CLASS_LABEL = 4

# %%
iris = pd.read_csv("iris.data", header=None)

os.makedirs('images/q3/p5/', exist_ok=True)

self_defined_T = {
    '1': np.array([-0.05, -1]),
    '2': np.array([-0.5, -5]),
    '3': np.array([-2, -5])
    }

for ((f1, f2), Ti) in [((1, 4), '1'), ((2, 3), '2'), ((3, 4), '3')]:
    zf1 = f1 - 1
    zf2 = f2 - 1
    transformed_matrix = iris[[zf1, zf2]].to_numpy() @ self_defined_T[Ti]
    x_axis = transformed_matrix
    y_axis = np.zeros(len(transformed_matrix))
    label = iris[CLASS_LABEL]
    df = pd.DataFrame({
        'x': x_axis,
        'y': y_axis,
        'label': label
    })

    fig = make_subplots(rows=2, cols=1)
    fig = px.histogram(df, x='x', color='label', labels={'label': 'Class'},
                       opacity=0.5, marginal="rug")
    fig.update_layout({
        'margin': {'l': 0, 'r': 0, 't': 40, 'b': 0},
        'title': {
            'text': f'Features {f1},{f2} Transformed with {self_defined_T[Ti]}',
            'x': 0.5
        },
        'barmode': 'overlay'
    })
    fig.show()
    fig.write_image(f'images/q3/p5/{f1}{f2}T{Ti}.png')