#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tabulate import tabulate

from pyifdm import methods
from pyifdm.methods import ifs
from pyifdm import weights as ifs_weights
from pyifdm import correlations as corrs
from pyifdm.helpers import rank, generate_ifs_matrix

import warnings

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True, precision=3)


# In[2]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate

from pymcdm import visuals

from pyifdm import methods
from pyifdm.methods import ifs
from pyifdm import weights as ifs_weights
from pyifdm import correlations as corrs
from pyifdm.helpers import rank

np.set_printoptions(suppress=True, precision=3)


# In[3]:


if_methods = {
    #if_proposed_method
    #IF-COPRAS
    'IF-COPRAS': methods.ifCOPRAS(),
    'IF-ARAS': methods.ifARAS(),
    'IF-TOPSIS': methods.ifTOPSIS(),
    'IF-EDAS': methods.ifEDAS(),
    'IF-MABAC': methods.ifMABAC(),
    'IF-MAIRCA': methods.ifMAIRCA(),
    'IF-MOORA': methods.ifMOORA(),
    'IF-CODAS': methods.ifCODAS(),
    'IF-VIKOR': methods.ifVIKOR()
    
}

method_names = if_methods.keys()


# In[4]:


matrix = np.array([
    [[0.234, 0.685], [1.0, 0], [0.167, 0.661], [
        0.801, 0.549], [0.085, 0.823], [0.26, 0.592]],
    [[0.18, 0.654], [0.241, 0.649], [0.097, 0.846],
        [1.0, 0], [0.15, 0.8], [1.0, 0]],
    [[0.254, 0.599], [0.197, 0.722], [0.187, 0.737], [
        0.131, 0.791], [0.36, 0.642], [0.256, 0.595]],
    [[0.26, 0.592], [0.304, 0.596], [0.142, 0.797], [
        0.04, 0.896], [0.171, 0.732], [0.142, 0.797]],
])

fuzzy_weights = np.array([[0.23, 0.659], [0.325, 0.518], [0.31, 0.436], [
                0.464, 0.255], [0.383, 0.493], [0.504, 0.732]])
types = np.array([1, -1, 1, -1, 1, 1])


# In[5]:


results = []
for name, method in if_methods.items():
    if name == 'IF-VIKOR':
        results.append(rank(method(matrix, fuzzy_weights, types)[1], descending=False))
    else:
        results.append(rank(method(matrix, fuzzy_weights, types)))


# In[6]:


print(tabulate(np.array(results).T, headers=list(if_methods.keys())))


# In[7]:


plt.rcParams['figure.figsize'] = (10, 5)
fig, ax = plt.subplots()

results = np.array(results)

spacer=0.7
high = int(np.ceil(np.max(results)))

plot_kwargs = dict(
    linewidth=4
)

for i in range(results.shape[1]):
    points = []
    markers = []
    for j in range(results.shape[0]):
        points.append((j - spacer/2, results[j, i]))
        points.append((j + spacer/2, results[j, i]))
        markers.append((j, results[j, i]))

    line, = ax.plot(*zip(*points), **plot_kwargs, label=f'$M_{{{i + 1}}}$')
    ax.plot(*zip(*markers), marker='o', c=line.get_color(), linestyle=' ')

ax.set_yticks(range(1, high + 1))
ax.set_yticklabels(range(1, high + 1), fontsize=10)
ax.set_ylabel('Ranking position', fontsize=15)

ax.set_xticks(range(results.shape[0]))
ax.set_xticklabels(method_names, fontsize=11)
ax.set_xlabel('Methods', fontsize=14)

ax.set_xlim([-0.5, results.shape[0] - 0.5])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        ncol=5, mode="expand", borderaxespad=0., fontsize=18)

ax.grid(alpha=0.5, linestyle='--')

fig.tight_layout()
plt.savefig('example1.pdf', dpi=400)


# In[8]:


M = np.zeros((results.shape[0], results.shape[0]))
for l1, c1 in enumerate(results):
    for l2, c2 in enumerate(results):
        M[l1, l2] = np.round(corrs.ws_rank_similarity_coef(c1, c2), 2)
M


# In[9]:


M


# In[10]:


plt.rcParams['figure.figsize'] = (10, 4)
fig, ax = plt.subplots()

results = np.array(results)
visuals.ranking_bar(results, list(method_names))

plt.savefig('example2.pdf', dpi=400)


# In[11]:


plt.rcParams['figure.figsize'] = (6, 6)
fig, ax = plt.subplots()

im = ax.imshow(M, cmap='Greens')
text_kwargs = dict(
    ha='center',
    va='center',
    color='b',
    fontsize=8
)
for i in range(len(method_names)):
    for j in range(len(method_names)):
        text = ax.text(j, i, '%0.2f' % M[i, j], **text_kwargs)

ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

plt.setp(ax.get_xticklabels(), rotation=45,
                 ha='left', rotation_mode='anchor')

ax.set_xticks(np.arange(len(method_names)), labels=method_names, fontsize=8)
ax.set_yticks(np.arange(len(method_names)), labels=method_names, fontsize=8)
plt.savefig('example1_corr.pdf', dpi=500)


# In[12]:


# real data matrix
matrix = np.array([
    [[0.47, 0.525], [0.47, 0.52], [0.29, 0.70], [0.43, 0.56]],
    [[0.53, 0.46], [0.55, 0.44], [0.63, 0.37], [0.59, 0.40]],
    [[0.43, 0.56], [0.40, 0.59], [0.42, 0.57], [0.36, 0.56]],
    [[0.52, 0.65], [0.08, 0.92], [0.67, 0.43], [0.91, 0.77]],
    [[0.25, 0.65], [0.48, 0.52], [0.67, 0.43], [0.03, 0.87]],
    [[0.68, 0.32], [0.23, 0.77], [0.42, 0.52], [0.44, 0.53]],
    [[0.03, 0.07], [0.09, 0.85], [0.01, 0.88], [0.69, 0.32]],
    [[0.27, 0.79], [0.33, 0.37], [0.81, 0.07], [0.86, 0.74]],
    [[0.41, 0.58], [0.43, 0.57], [0.42, 0.52], [0.44, 0.53]]
])

# randomly generated matrix
# 4 alternatives
# 10 criteria
random_matrix = generate_ifs_matrix(4, 10)
print(random_matrix)


# In[13]:


normalizations = {
    'Ecer normalization': ifs.normalization.ecer_normalization,
    'minmax_normalization': ifs.normalization.minmax_normalization,
    'Supriya normalization': ifs.normalization.supriya_normalization,
    'Swap normalization': ifs.normalization.swap_normalization,
}

types = np.array([1, -1, 1, -1])

for name, norm in normalizations.items():
    nmatrix = norm(matrix, types)
    print(f'{name}\n\n {nmatrix[:2]}\n')


# In[14]:


distances = {
    'Euclidean' : ifs.distance.euclidean_distance,
    'Grzegorzewski': ifs.distance.grzegorzewski_distance,
    'Hamming': ifs.distance.hamming_distance,
    'Luo distance': ifs.distance.luo_distance,
    'Normalized Hamming': ifs.distance.normalized_hamming_distance,
    'Normalized Euclidean': ifs.distance.normalized_euclidean_distance,
    'Wang Xin 1': ifs.distance.wang_xin_distance_1,
    'Wang Xin 2': ifs.distance.wang_xin_distance_2,
    'Yang Chiclana': ifs.distance.yang_chiclana_distance,
}

x = np.array([0.7, 0.3])
y = np.array([0.45, 0.5])

for name, distance in distances.items():
    if distance.__name__ == 'normalized_euclidean_distance':
        d = np.sqrt(1/2 * distance(x, y))
    elif distance.__name__ == 'normalized_hamming_distance':
        d = 1/2 * distance(x, y)
    else:
        d = distance(x, y)
    print(f'{name}: {d}')


# In[15]:


score_functions = {
    'Chen score 1': ifs.score.chen_score_1,                                                                                          
    'Chen score 2': ifs.score.chen_score_2,                                                                                          
    'Kharal score 1': ifs.score.kharal_score_1,                                                                                          
    'Kharal score 2': ifs.score.kharal_score_2,                                                                                          
    'Liu Wang score': ifs.score.liu_wang_score,                                                                                          
    'Supriya score': ifs.score.supriya_score,                                                                                          
    'Thakur score': ifs.score.thakur_score,                                                                                          
    'Wan Dong score 1': ifs.score.wan_dong_score_1,                                                                                          
    'Wan Dong score 2': ifs.score.wan_dong_score_2,                                                                                          
    'Wei score': ifs.score.wei_score,                                                                                          
    'Zhang Xu score 1': ifs.score.zhang_xu_score_1,                                                                                          
    'Zhang Xu score 2': ifs.score.zhang_xu_score_2,                                                                                          
}

x = np.array([0.9, 0.11])

for name, score in score_functions.items():
    d = score(x)
    print(f'{name}: {d}')


# In[16]:


weights_methods = {
    'Burillo Entropy': ifs_weights.burillo_entropy_weights,
    'Equal': ifs_weights.equal_weights,
    'Entropy': ifs_weights.entropy_weights,
    'Liu Entropy': ifs_weights.liu_entropy_weights,
    'Szmidt Entropy': ifs_weights.szmidt_entropy_weights,
    'Thakur Entropy': ifs_weights.thakur_entropy_weights,
    'Ye Entropy': ifs_weights.ye_entropy_weights,
}

for name, method in weights_methods.items():
    w = method(matrix)
    print(f'{name} \n {w}\n')


# In[17]:


matrix = np.array([
     [[0.467, 0.525], [0.47, 0.52], [0.29, 0.70], [0.43, 0.56]],
    [[0.53, 0.46], [0.55, 0.44], [0.63, 0.37], [0.59, 0.40]],
    [[0.43, 0.56], [0.40, 0.59], [0.42, 0.57], [0.36, 0.56]],
    [[0.52, 0.65], [0.08, 0.92], [0.67, 0.43], [0.91, 0.77]],
    [[0.25, 0.65], [0.48, 0.52], [0.67, 0.43], [0.03, 0.87]],
    [[0.68, 0.32], [0.23, 0.77], [0.42, 0.52], [0.44, 0.53]],
    [[0.03, 0.07], [0.09, 0.85], [0.01, 0.88], [0.69, 0.32]],
    [[0.27, 0.79], [0.33, 0.37], [0.81, 0.07], [0.86, 0.74]],
    [[0.41, 0.58], [0.43, 0.57], [0.42, 0.52], [0.44, 0.53]]
])
crisp_weights = np.array([0.5, 0.3, 0.15, 0.35, 0.44, 0.53 , 0.42, 0.5, 0.36, 0.56])
types = np.array([-1,1,1,1,-1,1,1,1,-1,1])


# In[18]:


crisp_weights = np.array([0.5, 0.3, 0.15, 0.35, 0.44, 0.53 , 0.42, 0.5, 0.36, 0.56])


# In[19]:


fuzzy_weights = np.array([[0.6, 0.35], [0.8, 0.2], [0.5, 0.4], [0.2, 0.7]])


# In[20]:


types = np.array([-1, 1, 1, -1])


# In[21]:


#IF_COPRAS
if_copras = methods.ifCOPRAS()
res = if_copras(matrix, fuzzy_weights, types)
print('PREFERENCES')
print(f'R: {res[0]}')
print(f'U: {res[1]}')
print(f'N: {res[2]}')
print(f'A: {res[3]}')
print(f'L: {res[4]}')
ranks = if_copras.rank()
print('RANKINGS')
print(f'R: {res[0]}')
print(f'U: {res[1]}')
print(f'N: {res[2]}')
print(f'A: {res[3]}')
print(f'L: {res[4]}')


# In[22]:


copras = {
    'Chen score 1': methods.ifCOPRAS(score=ifs.score.chen_score_1),                                                                                          
    'Chen score 2': methods.ifCOPRAS(score=ifs.score.chen_score_2),                                                                                          
    'Kharal score 1': methods.ifCOPRAS(score=ifs.score.kharal_score_1),                                                                                          
    'Kharal score 2': methods.ifCOPRAS(score=ifs.score.kharal_score_2),                                                                                          
    'Liu Wang score': methods.ifCOPRAS(score=ifs.score.liu_wang_score),                                                                                          
    'Supriya score': methods.ifCOPRAS(score=ifs.score.supriya_score),                                                                                          
    'Thakur score': methods.ifCOPRAS(score=ifs.score.thakur_score),                                                                                          
    'Wan Dong score 1': methods.ifCOPRAS(score=ifs.score.wan_dong_score_1),                                                                                          
    'Wan Dong score 2': methods.ifCOPRAS(score=ifs.score.wan_dong_score_2),                                                                                          
    'Wei score': methods.ifCOPRAS(score=ifs.score.wei_score),                                                                                          
    'Zhang Xu score 1': methods.ifCOPRAS(score=ifs.score.zhang_xu_score_1),                                                                                          
    'Zhang Xu score 2': methods.ifCOPRAS(score=ifs.score.zhang_xu_score_2),
}


# In[23]:


results = {}
for name, function in copras.items():
    results[name] = function(matrix, fuzzy_weights, types)


# In[24]:


print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
    headers=['Method'] + [f'M{i+1}' for i in range(matrix.shape[0])]))


# In[25]:


copras = {
    'Chen score 1': methods.ifCOPRAS(score=ifs.score.chen_score_1),                                                                                          
    'Chen score 2': methods.ifCOPRAS(score=ifs.score.chen_score_2),                                                                                          
    'Kharal score 1': methods.ifCOPRAS(score=ifs.score.kharal_score_1),                                                                                          
    'Kharal score 2': methods.ifCOPRAS(score=ifs.score.kharal_score_2),                                                                                          
    'Liu Wang score': methods.ifCOPRAS(score=ifs.score.liu_wang_score),                                                                                          
    'Supriya score': methods.ifCOPRAS(score=ifs.score.supriya_score),                                                                                          
    'Thakur score': methods.ifCOPRAS(score=ifs.score.thakur_score),                                                                                          
    'Wan Dong score 1': methods.ifCOPRAS(score=ifs.score.wan_dong_score_1),                                                                                          
    'Wan Dong score 2': methods.ifCOPRAS(score=ifs.score.wan_dong_score_2),                                                                                          
    'Wei score': methods.ifCOPRAS(score=ifs.score.wei_score),                                                                                          
    'Zhang Xu score 1': methods.ifCOPRAS(score=ifs.score.zhang_xu_score_1),                                                                                          
    'Zhang Xu score 2': methods.ifCOPRAS(score=ifs.score.zhang_xu_score_2),
}


# In[26]:


results = {}
for name, function in copras.items():
    results[name] = function(matrix, fuzzy_weights, types)


# In[27]:


print(tabulate([[name, *rank(pref)] for name, pref in results.items()],
    headers=['Method'] + [f'M{i+1}' for i in range(matrix.shape[0])]))


# In[28]:


results = np.array([rank(r) for r in list(results.values())])


# In[29]:


#Preferences_comparison
x = np.array([0.7, 0.8, 0.4, 0.6, 0.1])
y = np.array([0.6, 0.4, 0.7, 0.2, 0.2])

print(f'Pearson: {corrs.pearson_coef(x, y)}')
print(f'Spearman: {corrs.spearman_coef(x, y)}')


# In[30]:


#Rankings_comparison
x = np.array([2, 4, 9, 7, 5])
y = np.array([3, 2, 4, 9, 5])

print(f'Weighted Spearman: {corrs.weighted_spearman_coef(x, y)}')
print(f'WS rank similarity: {corrs.ws_rank_similarity_coef(x, y)}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




