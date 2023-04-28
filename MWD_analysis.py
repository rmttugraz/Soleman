#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)


# In[2]:


SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics


# In[4]:


df = pd.read_csv('data_complete_clean.csv', sep=',')


# - HD [mm]: hole depth
# - PR [dm/min]: penetration rate
# - HP [bar]: hammering pressure
# - FP [bar]: feed pressure = Hydraulic pressure that pushes the drill against the tunnel
# face.
# - DP [bar]: dumper pressure = Hydraulic pressure that absorbs the reaction force that
# the drill receives from the bedrock. The harder the bedrock, the greater the dumper
# pressure.
# - RS [r/min]: rotation speed
# - RP [bar]: rotation pressure
# - WF [l/min]: water flow
# - WP [bar]: water pressure
# - Time [hh:mm:ss]

# In[5]:


dfs = df[(df['Section number * 1000']>=555000)&(df['Section number * 1000']<=777000)].reset_index() 


# In[6]:


dfs


# In[7]:


dfs['Date and time at rockcontact'] = pd.to_datetime(dfs['Date and time at rockcontact'])


# In[8]:


selected_columns_0 = [
    #'Unnamed: 0', 
    #'Unnamed: 0.1', 
    'HD mm', 
    'PR dm/min', 
    'HP bar', 
    'FP bar',
    'DP bar', 
    'RS r/min', 
    'RP bar', 
    'WF l/min', 
    'WP bar', 
    #'Time',
    #'reference', 
    #'Hole number', 
    'Hole type', 
    #'Date and time at rockcontact',
    #'Boom', 
    'Section number * 1000', 
    #'x\ty\tz\tmm',
    #'Lookout\tLookoutdirection(Degrees*10)\tsample interval(cm)',
    #'Rig serial number', 
    #'dir', 
    'number', 
    #'reference',
    #'file', 
    #'x mm', 
    #'y mm', 
    #'z mm',
]


# # ensure all numbers are numerical data

# In[9]:


for c in selected_columns_0:
    dfs[c] = dfs[c].astype(str).str.replace(r"_", '').astype(float)
selected_columns = selected_columns_0 + ['Date and time at rockcontact']

datas = dfs[selected_columns]


# # average data for a single borehole

# In[10]:


datas_avg = pd.DataFrame()
for t in datas['Date and time at rockcontact'].unique():
    tmp = datas[datas['Date and time at rockcontact'] == t].reset_index(drop=True)
    datas_avg = datas_avg.append(tmp[selected_columns_0].mean(), ignore_index=True).reset_index(drop=True)
datas = datas_avg.copy()
datas


# # now the MWD data is clean and ready to add target variables

# In[11]:


datas


# # add targets

# In[12]:


expl = pd.read_csv('explosives.csv')


# In[13]:


expl


# In[14]:


expl.columns


# # ensure all numbers are numerical data

# In[15]:


expl_columns_tonum = ['area\n[m2]', 'progression\n[m]',
       'excavation \nvolume\n[m3]', 'wg3,piece', 'wg3,kg',
       'wg25,piece', 'wg25,kg', 'wg,kg', 'anfo,kg', 'f06,piece',
       'fconnect,piece', 'eldet,piece', 'pyro1', 'pyro2', 'pyro3', 'pyro4',
       'pyro5', 'pyro6', 'pyro7', 'pyro8', 'pyro9', 'pyro10', 'pyro12',
       'pyro14', 'pyro16', 'pyro17', 'pyro18', 'pyro19', 'pyro20', 'pyro21',
       'total', 'kg/m3', 'faceN']


# In[16]:


expl[expl_columns_tonum] = expl[expl_columns_tonum].apply(pd.to_numeric, errors='coerce')
expl.date = pd.to_datetime(expl.date)
expl = expl.dropna().reset_index(drop=True)


# # add target values to MWD data

# In[17]:


target_columns = ['excavation \nvolume\n[m3]', 'total','kg/m3']
datas[target_columns] = 0
for exp in expl.faceN.unique():
    for col in target_columns:
        val = sum(expl[expl.faceN == exp][col])
        datas.loc[datas.number == exp,col]=val


# # now the dataset is ready for analysis

# In[18]:


datas


# # PCA 

# In[19]:


#features = ['HD mm','PR dm/min','HP bar','FP bar','DP bar','RS r/min','RP bar','WF l/min','WP bar']
features = ['PR dm/min','HP bar','FP bar','DP bar','RS r/min','RP bar','WF l/min','WP bar']
#features = ['PR dm/min','HP bar','RS r/min']

X = datas.loc[:, features].values
            
x = StandardScaler().fit_transform(X)

components = len(features)    
pca = PCA(n_components=components)

principalComponents = pca.fit_transform(x)
print(pca.explained_variance_ratio_)


# In[20]:


pc = pd.DataFrame(principalComponents)


# # add PCA results to main dataset

# In[21]:


rf_datas = datas.join(pc)
rf_datas


# In[ ]:





# # Random forest

# In[22]:


from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.ensemble import RandomForestRegressor # for building the model


# # reshuffle the data

# In[23]:


rf_datas = rf_datas.sample(frac=1).reset_index(drop=True)
rf_datas


# # test train split

# In[24]:


columns = [ 'Hole type','FP bar', 'DP bar', 'RS r/min','RP bar', 'WF l/min', 
           'PR dm/min', 'HP bar','HD mm']+[0,1,2]


# In[25]:


x = rf_datas[columns] # Features
y = rf_datas['kg/m3'].values  # Target


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# # train RF

# In[27]:


rf = RandomForestRegressor(n_estimators = 600, random_state = 42)
rf.fit(x_train, y_train)


# # Predict with RF and evaluate

# In[28]:


prediction = rf.predict(x_test)
mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
print(mse,rmse)


# In[ ]:





# # feature importance

# In[29]:


importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
forest_importances = pd.Series(importances, index=columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")


# # visual check on pred and act

# In[30]:


rf_out = pd.DataFrame(prediction, columns=['pred'])
rf_out['act'] = y_test
rf_out = rf_out.sort_values(by=['act']).reset_index(drop=True)
rf_out


# In[31]:


plt.plot(rf_out.pred, color='red')
plt.plot(rf_out.act)


# In[ ]:




