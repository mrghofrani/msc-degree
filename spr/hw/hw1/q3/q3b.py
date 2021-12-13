import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.options.plotting.backend = "plotly"


CLASS_LABEL = 4

iris = pd.read_csv("iris.data", header=None)


T = {
    '1': np.array([1.2, -0.3]),
    '2': np.array([-1.8, 0.6]),
    '3': np.array([1.4, 0.5]),
    '4': np.array([-0.5, -1])
}

os.makedirs('images/q3/p2/', exist_ok=True)
for (f1, f2) in [(1, 2), (1, 3), (2, 4)]:
    for Ti in T.keys():
        zf1 = f1 - 1
        zf2 = f2 - 1
        transformed_matrix = iris[[zf1, zf2]].to_numpy() @ T[Ti]
        x_axis = transformed_matrix
        y_axis = np.zeros(len(transformed_matrix))
        label = iris[CLASS_LABEL]
        df = pd.DataFrame({
            'x': x_axis,
            'y': y_axis,
            'label': label
        })

        fig = make_subplots(rows=2, cols=1)
        fig = px.histogram(df, x='x', color='label', opacity=0.5,  marginal="rug")
        fig.update_layout({
            'margin': {'l': 0, 'r': 0, 't': 40, 'b': 0},
            'title': {
                'text': f'Features {f1},{f2} Transformed with T{Ti}',
                'x': 0.5
            },
            'barmode': 'overlay'
        })
        fig.show()
        fig.write_image(f'images/q3/p2/{f1}{f2}T{Ti}.png')
