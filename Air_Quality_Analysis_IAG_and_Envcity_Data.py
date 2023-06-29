#!/usr/bin/env python
# coding: utf-8

# ## Lendo os dados e Configurações do Matplotlib
# 

# In[11]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sys import version

from IPython.display import display, HTML

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.sans-serif": "Times",
    "font.size" : 10,
})

print("python version: ", version)
print("Numpy version: ", np.__version__)
print("Pandas version: ", pd.__version__)


# In[12]:


# !wget https://media.githubusercontent.com/media/MateusMaruzka/aqm-dataset/main/envcity_aqm_df.csv -O aqm.csv


# In[13]:


#!wget https://raw.githubusercontent.com/MateusMaruzka/aqm_envcity_sw/main/alphasense_sensors.py -O alphasense_sensors.py
#!wget https://raw.githubusercontent.com/MateusMaruzka/aqm_envcity_sw/main/dados_correcao_temp.py -O dados_correcao_temp.py
#!wget https://raw.githubusercontent.com/MateusMaruzka/aqm_envcity_sw/main/dados_alphasense.py -O dados_alphasense.py


# ## Organizando o dataset

# In[14]:

aqm = pd.read_csv('envcity_aqm_df.csv')

#%%

from itertools import product

for label, p, s in product(['anem'], ['e1_', 'e2_', 'e2sp_'], ['_volt', '']):
    print(p + label+s)
    try:
        aqm.drop(labels = p + label + s, axis = 1,inplace=True)
    except:
        print('err')
# aqm.set_index('time', inplace=True)

# In[34]:

aqm_filtered = aqm.copy()

for index, row in aqm.iterrows():
    if (row == -3).sum() > 0:
        aqm_filtered.drop(index, inplace=True)
# In[36]:

# aqm_filtered.reset_index(drop=True)
aqm_filtered.set_index('time', inplace=True)
aqm = aqm_filtered

#%%
labels =  ['co', 'so2', 'ox', 'no2']
prefix = ['e1_', 'e2_', 'e2sp_']
suffix = ['_ae', '_we']

for label, p, s in product(labels, prefix, suffix):
  data = aqm[p+label+s]
  idx = (data > 6.14) | (data < 0.05)
  data.loc[idx] = np.nan

for p in prefix:
  data = aqm[p + 'temp']
  idx = (data > 50) | (data <= 1)
  data.loc[idx] = np.nan
  
# In[37]:


# aqm.describe()


# ## Métricas de Avaliação
# 
# * Métricas de avaliação [https://amt.copernicus.org/articles/11/291/2018/amt-11-291-2018.pdf]
# 1. $R^2$
# 
# 2. **Pearson r**
# 
# 3. $\mathrm{RMSE} = \sqrt{\frac{\sum_{i=1}^{N}(y_{ref} - \hat{y})^{2})}{N}}$
# 
# 4. $\mathrm{CvMAE} = \frac{\overbrace{MAE}^{\textrm{Mean absolute error}}}{\mu_{ref}} = \frac{1}{\mu_{ref}} \frac{\sum_{i=1}^{N}{|y_{ref} - \hat{y}}|}{N}$
# 

# In[38]:


def pearson_r(y, yref):
    my = np.mean(y)
    myref = np.mean(yref)
    _y = y - my
    _yref = yref - myref
    num = np.sum(np.dot(_y, _yref))
    den = np.sum(_y**2) * np.sum(_yref**2)
    den = np.sqrt(den)

    return num/den

def mse(y, yref):
    return np.mean(np.square(np.subtract(yref, y)))

def rmse(y, yref):
    return np.sqrt(mse(y, yref))

def mae(y, yref):
    return np.mean(np.abs(np.subtract(yref, y)))

def cvmae(y, yref):
    yref_mean = np.mean(yref)
    return mae(y, yref) / yref_mean


# In[39]:


def exploratory_analysis(dict_data_e1, dict_data_e2, labels, latex_labels, start, end):

    table_exploratory_analysis = {}

    for idx, l in enumerate(labels):

        e1 = dict_data_e1[l]
        e2 = dict_data_e2[l]

        concatenated = pd.concat([e1, e2], axis=1, keys=['Station 1', 'Station 2'])
        table_exploratory_analysis[l] = describe(concatenated, ['median'], ['25%', '50%', '75%'])

    return table_exploratory_analysis

# ### Funções Gráficos
# 
# **plot_data_by_time_and_regr_plot** gera dois gráficos, um do lado do outro. O primeiro é os dados em função do tempo e o segundo é um gráfico de dispersão entre os dois sensores

# In[40]:


import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import EngFormatter

def plot_data_by_time_and_regr_plot(dict_data_e1, dict_data_e2, labels, latex_labels, start, end, e1_label = 'Station 1', e2_label = 'Station 2', style_plot = 'dark'):

    for idx, l in enumerate(labels):

        e1 = dict_data_e1[l].loc[start:end]
        e2 = dict_data_e2[l].loc[start:end]

        concatenated = pd.concat([e1, e2], axis=1, keys=[e1_label, e2_label])

        #with sns.axes_style(style=style_plot):
        sns.set_palette(style_plot)

        # fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        fig = plt.figure(figsize=(12, 4), layout='tight')
        gs = GridSpec(1, 2, width_ratios=[3,1])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        sns.lineplot(data=concatenated, linewidth=1, ax=ax1)
        sns.regplot(data=concatenated, x = e1_label, y = e2_label, ax = ax2, marker = '.')

        # Set axis labels and title
        ax1.set_title(f"$\mathrm{{{latex_labels[idx]}}}$ Concentration from {start} to {end}")
        ax1.set_ylabel("Concentration (ppb)")
        ax1.set_xlabel("Date")

        ax2.set_ylabel(e2_label)
        ax2.set_xlabel(e1_label)

        ax2.set_box_aspect(1)

        # Format the x-axis labels
        date_format = mdates.DateFormatter('%d/%m/%y')
        ax1.xaxis.set_major_formatter(date_format)
        ax1.tick_params(axis='x', labelrotation=45)

        # ax1.set_xlim([0, dict_data_e1[l].index[-1]])

        # Using metric prefixes
        ax1.yaxis.set_major_formatter(EngFormatter())
        ax2.yaxis.set_major_formatter(EngFormatter())
        ax2.xaxis.set_major_formatter(EngFormatter())

        # ax1.margins(0.1, 0.1)
        # ax2.margins(0.2, 0.2)

        ax1.autoscale(enable = None, axis="x", tight=True)
        # ax2.autoscale(enable=None, axis="x", tight=True)

        # ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.25, 1.05))
        # fig.savefig(f"{l}_time.pdf", format='pdf', dpi=fig.dpi, bbox_inches='tight')

        plt.show()

def plot_data_by_time(dict_data_e1, dict_data_e2, labels, latex_labels, start, end, style_plot = 'dark'):

    for idx, l in enumerate(labels):

        e1 = dict_data_e1[l].loc[start:end]
        e2 = dict_data_e2[l].loc[start:end]

        concatenated = pd.concat([e1, e2], axis=1, keys=['Station 1', 'Station 2'])

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sns.lineplot(data=concatenated, linewidth=1, ax=ax)

        # Set axis labels and title
        ax.set_title(f"$\mathrm{{{latex_labels[idx]}}}$ Concentration from {start} to {end}")
        ax.set_ylabel("Concentration (ppb)")
        ax.set_xlabel("Date")

        # Format the x-axis labels
        date_format = mdates.DateFormatter('%d/%m/%y')
        ax.xaxis.set_major_formatter(date_format)
        ax.tick_params(axis='x', labelrotation=45)

        # format the y-axis labels
        ax.yaxis.set_major_formatter(EngFormatter())

        # ax.margins(0.2, 0.2)
        ax.autoscale(enable=None, tight=True)

        ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.3, 1.05))
        fig.savefig(f"{l}_time.pdf", format='pdf', dpi=fig.dpi, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

def plot_boxplot(dict_data_e1, dict_data_e2, labels, latex_labels, start, end, style_plot = 'dark'):

    for idx, l in enumerate(labels):

        e1 = dict_data_e1[l].loc[start:end]
        e2 = dict_data_e2[l].loc[start:end]

        # For comparing all equations just uncomment these two lines
        # e1_concat = pd.concat(e1, axis = 1, keys = ["Equação 1", "Equação 2", "Equação 3", "Equação 4"])
        # e2_concat = pd.concat(e2, axis = 1, keys = ["Equação 1", "Equação 2", "Equação 3", "Equação 4"])
        concatenated = pd.concat([e1, e2], axis=1, keys=['Station 1', 'Station 2'])

        melted = pd.melt(concatenated, var_name=['Data', 'sensor'], value_name='value', ignore_index=False)

        sns.set_palette(style_plot)
        #with sns.axes_style(style=style_plot):

        fig, ax = plt.subplots(1, 1, figsize = (5, 2))
        bp = sns.boxplot(data=melted, x="sensor", y = 'value', hue = 'Data', ax = ax, width=0.7, linewidth=1, fliersize=5)
        bp.set(xlabel = "")
        ax.set_title(f"$\mathrm{{{latex_labels[idx]}}}$ Concentration from {start} to {end}", fontsize = 10)
        ax.set_ylabel("ppb")
        ax.legend(loc='upper right', frameon = False, bbox_to_anchor=(1.35, 1.05))
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)
        ax.yaxis.set_major_formatter(EngFormatter())

        fig.savefig(f"{l}.pdf", format = 'pdf', dpi=fig.dpi, bbox_inches =  'tight')
        plt.show()

def plot_boxplot_alphasense_equations(dict_data_e1, dict_data_e2, labels, latex_labels, start, end, style_plot = 'dark'):

    for idx, l in enumerate(labels):

        e1 = dict_data_e1[l].loc[start:end]
        e2 = dict_data_e2[l].loc[start:end]

        concatenated = pd.concat([e1, e2], axis=1, keys=['Station 1', 'Station 2'])

        melted = pd.melt(concatenated, var_name=['Data', 'sensor'], value_name='value', ignore_index=False)

        sns.set_palette(style_plot)
        #with sns.axes_style(style=style_plot):

        fig, ax = plt.subplots(1, 1, figsize = (5, 2))
        bp = sns.boxplot(data=melted, x="sensor", y = 'value', hue = 'Data', ax = ax, width=0.7, linewidth=1, fliersize=3)
        ax.set_title(f"$\mathrm{{{latex_labels[idx]}}}$ Concentration from {start} to {end}", fontsize = 10)
        ax.set_ylabel("ppb")
        ax.legend(loc='upper right', frameon = False, bbox_to_anchor=(1.4, 1.05))
        ax.yaxis.set_major_formatter(EngFormatter())

        fig.savefig(f"{l}.pdf", format = 'pdf', dpi=fig.dpi, bbox_inches =  'tight')
        plt.show()


# ### Análise

# #### Resultados complementas:
# 1. Comunicação degrada a partir de 30 graus
# 2. Pode estar relacionada ao CI SX1276
# 
# Ações:
# Dropar toda as linhas com erro

# In[41]:


labels =  ['so2_we', 'so2_ae']
preffix = ['e2sp_']

#latex_labels it is only for printing
latex_labels = ['CO', 'NO_2', 'O_X', 'SO_2', 'PM1.0', 'PM2.5', 'PM10']


# In[42]:

# for label in labels:
#   for p in preffix:
#     plt.figure()
    
#     data_filtered = aqm[p+label]
        
#     data_temp = aqm[p+'temp']
    
#     data_filtered = data_filtered.dropna()
#     data_temp = data_temp.dropna()
    
#     teste = pd.concat([data_filtered, data_temp], axis =1)
    
#     plt.gca().scatter(y = teste[p+label], x=teste[p+'temp'], marker = '.')
#     plt.gca().set_xlabel('Temperatura')
#     plt.gca().set_ylabel('Tensão')
#     plt.title(label)

#     plt.show()
#     print(aqm[p+label].min(), aqm[p+label].max())

#%%
# ['2023-03-18 10:00:00':'2023-03-22 10:00:00'].
plt.figure()
aqm['e2sp_co_we'].loc['2023-03-18 10:00:00':'2023-03-22 10:00:00'].plot(marker = '.', linewidth = 0.1, color = 'b')
plt.gca().set_ylim([-0.5, 1])
ax = plt.gca().twinx()
aqm['e2sp_umid'].loc['2023-03-18 10:00:00':'2023-03-22 10:00:00'].plot(marker = '.', linewidth = 0.1, ax = ax, color = 'r')
ax.set_ylim([0, 99])
# plt.gcf().autofmt_xdate()
plt.show()

#%%

from sklearn.preprocessing import StandardScaler, MinMaxScaler, KernelCenterer,Normalizer

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV,\
                                    validation_curve, cross_val_score, \
                                    cross_validate, \
                                    RepeatedKFold, cross_val_predict, StratifiedKFold
from sys import version


#%%

labels =  ['co_we', 'co_ae']
preffix = ['e2sp_']
label_ref= 'iag_co'

df = aqm.loc['2023-03-18 10:00:00':'2023-03-22 10:00:00']

df = df[[preffix[0] + labels[0], preffix[0] + labels[1],
         'e2sp_so2_we','e2sp_so2_ae', 'e2sp_temp', label_ref]]

df.index = pd.to_datetime(df.index)
df.dropna(inplace=True)

Yco = df['e2sp_temp']

Xco = df.loc[Yco.index][[preffix[0] + labels[0], preffix[0] + labels[1],'e2sp_so2_we','e2sp_so2_ae']]

X_train, X_valid, y_train, y_valid = train_test_split(Xco, Yco, train_size=0.7)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

kfold = RepeatedKFold(n_splits = 5, n_repeats = 1)
# kfold = StratifiedKFold(n_splits = 5)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

#%%

# param_grid = {"randomforestregressor__n_estimators": [15, 50],
#               "randomforestregressor__max_depth": [64, 512, 1024],
#               # "randomforestregressor__oob_score" : [True],
#               # "randomforestregressor__bootstrap" : [True],
#               'randomforestregressor__max_features': ["sqrt", "log2", 0.3, 0.1],
#               'randomforestregressor__criterion': ['squared_error']}
from scipy.stats import uniform, randint

param_grid = {"randomforestregressor__n_estimators": randint(1, 512),
              "randomforestregressor__max_depth": randint(1, 512),
              #  "randomforestregressor__oob_score" : [True],
              "randomforestregressor__bootstrap" : [False, True],
              'randomforestregressor__max_features': ["sqrt", "log2", None],
              'randomforestregressor__criterion': ['squared_error', 'absolute_error', 'friedman_mse']}
# 
regressor = make_pipeline( RandomForestRegressor())
# gs = MLPRegressor(2, max_iter = 10000)

gs = RandomizedSearchCV(regressor, param_distributions=param_grid, n_jobs=-1, verbose = 3,\
                  return_train_score=True, cv = kfold, error_score = 'raise')

    
res = gs.fit(X_train,y_train)

#%%
print(res.cv_results_)
#%%

# r2_score(y_true, y_pred)
# x = X_train[:, 0]
# print("Sem regr", r2_score(x, y_train))
print("Train Score: ", gs.score(X_train, y_train))
print("Test Score: ", gs.score(X_test, y_test))
print("Validation Score: ", r2_score(y_valid, gs.predict(X_valid)))

sns.regplot(x = y_valid, y = gs.predict(X_valid))
sns.regplot(x = y_test, y = gs.predict(X_test))


#%%

plt.plot(df['e2sp_temp'].values)
plt.plot(gs.predict(Xco))