
import os
import matplotlib.pyplot as plt
import pandas as pd

path = os.getcwd()
E = pd.read_csv(path + '/AE_BCE_lambda1_E_w1_bs2000_2d.csv',
                index_col='Unnamed: 0')
R = pd.read_csv(path + '/AE_BCE_lambda1_R_w1_bs2000_2d.csv',
                index_col='Unnamed: 0')

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()
ax.scatter(E['Z0'], E['Z1'], c='Red', s=30, marker='o', label='E')
for j, txt in enumerate(E.index.tolist()):
    print(j)
    ax.text(tuple(E['Z0'].head(j+1))[j], tuple(E['Z1'].head(j+1))[j], txt,
            size=8, c='Red')

ax.scatter(R['Z0'], R['Z1'], c='black', s=30, marker='x', label='R')
for j, txt in enumerate(R.index.tolist()):
    ax.text(tuple(R['Z0'].head(j+1))[j], tuple(R['Z1'].head(j+1))[j], txt,
            size=8, c='black')
ax.legend()
plt.show()