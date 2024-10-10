import pandas as pd
import numpy as np
import plotly.express as px

pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True

df = pd.read_csv("../dataset.csv")
print(df.columns)