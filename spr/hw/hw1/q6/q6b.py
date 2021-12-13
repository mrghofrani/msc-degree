import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import multivariate_normal

pd.options.plotting.backend = "plotly"

X_RANGE = 105
Y_RANGE = 68

dataset = pd.read_csv('first_half_logs.csv')
dataset.columns = ['timestamp', 'tag_id', 'x_pos',
                   'y_pos', 'heading', 'direction',
                   'energy', 'speed', 'total_distance']

mean_location = dataset.groupby(['tag_id']).mean()[['x_pos', 'y_pos']]
covariance_location = dataset[['tag_id', 'x_pos', 'y_pos']].groupby(['tag_id']).cov()

os.makedirs('images/q6/partb/', exist_ok=True)

covariance_location = dataset[['tag_id', 'x_pos', 'y_pos']].groupby(['tag_id']).cov()

x = np.arange(0, X_RANGE, 1)
y = np.arange(0, Y_RANGE, 1)

for player in pd.unique(dataset['tag_id'].sort_values()):
    mean = mean_location.loc[player].to_numpy()
    cov = covariance_location.loc[player].to_numpy()
    dist = multivariate_normal(cov=cov, mean=mean)
    yy, xx = np.mgrid[0:X_RANGE:1, 0:X_RANGE:1]
    pos = np.dstack((xx, yy))
    z = dist.pdf(pos)

    with np.printoptions(precision=3, suppress=True):
        fig = go.Figure(
            data=go.Contour(x=x, y=y, z=z, colorscale="bugn"),
            layout={
                'xaxis': {
                    'range': [0, X_RANGE-1],
                    'dtick': 10
                },
                'yaxis': {
                    'range': [0, Y_RANGE-1],
                    'dtick': 10
                },
                'title': {
                    'text': f'Player {player} <br> Mean={mean}, Covariance={cov}',
                    'x': 0.5
                },
                'margin': {
                    'b': 0,
                    'l': 0,
                    'r': 0,
                    't': 50
                }})
        fig.show()
        fig.write_image(f'images/q6/partb/{player}.png')