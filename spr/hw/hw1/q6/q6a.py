
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import multivariate_normal

pd.options.plotting.backend = "plotly"
# %% Contants

X_RANGE = 105
Y_RANGE = 68

# %% Preparing Data

dataset = pd.read_csv('first_half_logs.csv')
dataset.columns = ['timestamp', 'tag_id', 'x_pos',
                   'y_pos', 'heading', 'direction',
                   'energy', 'speed', 'total_distance']
# %% Part a
os.makedirs('images/q6/parta/', exist_ok=True)
mean_location = dataset.groupby(['tag_id']).mean()[['x_pos', 'y_pos']]
mean_location_bar_plot_figure = mean_location.plot.bar(barmode='group')
mean_location_bar_plot_figure.update_layout({
    'margin': {'l': 0, 'r': 0, 't': 30, 'b': 0},
    'xaxis': {
        'dtick': 1
    },
    'title': {
        'text': f'Mean Location of Each Player',
                'x': 0.5
    }
})
mean_location_bar_plot_figure.write_image('images/q6/parta/plot.png')
mean_location_scatter_plot_figure = go.Figure(
    data=go.Scatter(
        x=mean_location['x_pos'], y=mean_location['y_pos'], mode="markers+text", text=mean_location.index,
        textposition="top center"
        ),
    layout={
        'xaxis': {
            'range': [0, X_RANGE+10],
            'dtick': 10
        },
        'yaxis': {
            'range': [0, Y_RANGE+10],
            'dtick': 10
        }})
mean_location_scatter_plot_figure.show()
mean_location_scatter_plot_figure.write_image(f'images/q6/parta/scatter.png')
