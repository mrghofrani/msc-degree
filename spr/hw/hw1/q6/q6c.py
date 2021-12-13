
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

randomly_select_player = np.random.choice(pd.unique(dataset['tag_id']), 3, replace=False)
xx, yy = np.mgrid[0:X_RANGE:1, 0:Y_RANGE:1]
pos = np.dstack((xx, yy))

for player in randomly_select_player:
    mean = mean_location.loc[player].to_numpy()
    cov = covariance_location.loc[player].to_numpy()
    dist = multivariate_normal(cov=cov, mean=mean)
    print(f"Player {player} --------")
    for _ in range(3):
        i_random = np.random.randint(105)
        j_random = np.random.randint(68)
        selected_pos = pos[i_random][j_random]
        probability = dist.pdf(selected_pos)
        print("Position", selected_pos, end=' ')
        print("Probability", probability)