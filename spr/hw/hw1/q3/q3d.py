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

os.makedirs('images/q3/p4/', exist_ok=True)
for (f1, f2) in [(1, 4), (2, 3), (3, 4)]:
    zf1 = f1 - 1
    zf2 = f2 - 1
    fig = px.scatter(iris, x=zf1, y=zf2, color=CLASS_LABEL,
                     labels={f'{zf1}': f'Feature {f1}',
                             f'{zf2}': f'Feature {f2}',
                             f'{CLASS_LABEL}': f"Class"})
    fig.update_layout(margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
    fig.show()
    fig.write_image(f'images/q3/p4/{f1}{f2}.png')