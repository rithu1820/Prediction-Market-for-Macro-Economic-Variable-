#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


pip install pmdarima


# In[3]:


import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import gmean
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# In[4]:


import warnings
warnings.filterwarnings(action="ignore")


# In[5]:


df= pd.read_csv("new_dataset.csv")


# In[6]:


df.head()


# In[7]:


df['date']=pd.to_datetime(df['date'])
df


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.plot('date','unemp_rate')


# In[11]:


df.corr()


# In[12]:


df['unemp_rate'].corr(df['int_rate'])


# In[13]:


df['unemp_rate'].corr(df['GDP'])


# In[14]:


df['unemp_rate'].corr(df['weekly_earnings'])


# In[15]:


df['unemp_rate'].corr(df['yield'])


# In[16]:


x_cor=df.corr()


# In[17]:


plt.figure(figsize=(11,8))
dataplot = sns.heatmap(x_cor, cmap="binary_r", annot=True)
plt.show()


# In[18]:


col_names = ['GDP', 'unemp_rate', 'weekly_earnings', 'int_rate', 'yield']
X = df[col_names]


# In[19]:


fd = df.drop("date",1)


# In[20]:


fd_1=fd.drop("unemp_rate",1)


# In[21]:


fd


# In[22]:


scaler = StandardScaler().fit(X.values)
X = scaler.transform(X.values)
X = pd.DataFrame(X, columns = col_names)
X


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams["figure.figsize"] = (10,6)


# In[24]:


plt.plot(df['date'], df['GDP'], label='GDP',linewidth=2)
plt.xlabel('Years')
plt.ylabel('GDP')
plt.title("UK's Country GDP")
plt.legend()
plt.show()


# In[25]:


df.plot('date','unemp_rate')
plt.xlabel('Years')
plt.ylabel('Employment_status')
plt.title("Employment status in UK")
plt.legend()
plt.show()


# In[26]:


df_norm = X / X.max(axis=0)
df_norm


# In[27]:


df_norm.plot.line(subplots=True)


# In[28]:


features = ['GDP', 'unemp_rate', 'weekly_earnings', 'yield', 'int_rate']
x = df_norm.loc[:, features].values


# In[29]:


x


# In[30]:


y_cred = df_norm.loc[:,['unemp_rate']].values


# In[31]:


y_cred


# In[32]:


pd.DataFrame(data = x, columns = features).head()


# In[33]:


x.shape


# In[34]:


from sklearn.decomposition import PCA
pca= PCA()
principalComponents = PCA().fit(x)


# In[35]:


np.round((principalComponents.explained_variance_ratio_*100),1)


# In[36]:


pca = PCA(n_components=3)
pca.fit(x)
columns = ['Pca_%i' % i for i in range(3)]
df_pca = pd.DataFrame(pca.transform(x), columns=columns, index=pd.DataFrame(x).index)
df_pca.head()


# In[37]:


df_pca.shape


# In[38]:


pca.explained_variance_ratio_


# In[39]:


# Visualize the PCA & it's explained variance.
plt.rcParams['figure.figsize'] = (8,6)
fig, ax = plt.subplots()
xi = np.arange(2, 5, step=1)
yi = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi,yi,marker='*',linestyle='--',color='black')

plt.xlabel('Number of PCA components')
plt.xticks(np.arange(1,6, step=1))
plt.ylabel('Cumulative variance (%)')
plt.title('PCA components needed to explain variance in the data')

plt.axhline(y=0.95, color='green',linestyle='--')
plt.text(1, 0.9, '95% threshold value',fontsize=12)
plt.axhline(y=1,color='red',linestyle='-')
plt.text(1, 1.02, '100% explained variance',fontsize=12)

ax.grid(axis='x')
plt.show()


# In[40]:


df[['unemp_rate']].head()


# In[41]:


finalDf = pd.concat([df_pca, df[['unemp_rate']]], axis = 1)
finalDf


# In[42]:


df[['unemp_rate']].head()
finalDf = pd.concat([df_pca, df['unemp_rate']], axis = 1)
finalDf


# In[43]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
finalDf


# In[44]:


plt.figure(figsize=(10,6))
sns.heatmap(finalDf.corr(), annot=True)


# In[45]:


Pca_0 = finalDf['Pca_0']
Pca_1 = finalDf['Pca_1']
Pca_2 = finalDf['Pca_2']


# In[46]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# In[47]:


plt.figure(figsize=(10,10))
plt.scatter(Pca_0, Pca_1, c=df['unemp_rate'],cmap='plasma')
plt.xlabel('Pca0')
plt.ylabel('Pca1')


# In[48]:


plt.figure(figsize=(10,10))
plt.scatter(Pca_0, Pca_2,c=df['unemp_rate'],cmap='plasma')
plt.xlabel('Pca0')
plt.ylabel('Pca2')


# In[49]:


plt.figure(figsize=(10,10))
plt.scatter(Pca_1, Pca_2,c=df['unemp_rate'],cmap='plasma')
plt.xlabel('Pca1')
plt.ylabel('Pca2')


# In[50]:


df_cov = finalDf.cov()
df_cov.head()


# In[51]:


cov_mat = np.cov(finalDf.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[52]:


train_data, test_data = df.unemp_rate[3:int(len(df.unemp_rate)*0.67)], df.unemp_rate[int(len(df.unemp_rate)*0.67):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('date')
plt.ylabel('unemp_rate')
plt.plot(df.unemp_rate, 'red', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()


# In[53]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = finalDf.drop(['unemp_rate'], axis = 1)
y = finalDf['unemp_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.67, random_state = 0)
scaling = LinearRegression()
scaling.fit(X_train,y_train)
intercept = scaling.intercept_
print(intercept)
coefficent = scaling.coef_
print(coefficent)
r2 = scaling.score(X_test, y_test)
print(r2*100)
ypred=scaling.predict(X_test)


# In[54]:


ypred


# In[55]:


y_test


# In[56]:


x = np.arange(121)
y1 = ypred
y2 = y_test 
plt.plot(x, y1, label ='predictions')
plt.plot(x, y2, label ='actual_data') 
plt.legend()
plt.show()


# In[57]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, ypred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, ypred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, ypred)))
print('Accuracy', metrics.r2_score(y_test, ypred))


# In[ ]:





# In[58]:


import statsmodels.api as sm
X=df.drop(["unemp_rate", "date"], 1)
y=df['unemp_rate']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[59]:


plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('date')
plt.ylabel('unemp_rate')
plt.plot(df['date'], df['unemp_rate'])
plt.title('Time Series')
plt.show()


# In[60]:


pip install --upgrade numpy


# In[61]:


import numpy as np


# In[62]:


pip install pmdarima


# In[63]:


import pmdarima as pm


# In[64]:


from statsmodels.tsa.stattools import adfuller


# In[65]:


def adfuller_test(unemp_rate):
    result=adfuller(unemp_rate)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")
        
adfuller_test(df['unemp_rate'])


# In[66]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


# In[67]:


finalDf = pd.concat([finalDf, df[['date']]], axis = 1)
finalDf


# In[68]:


fd2=finalDf.drop("Pca_0", axis=1)
fd2_1=fd2.drop("Pca_1", axis=1)
fd2_2=fd2_1.drop("Pca_2", axis=1)
fd2_2


# In[69]:


fd2_2['unemp_rate First Difference'] = fd2_2['unemp_rate'] - fd2_2['unemp_rate'].shift(1)


# In[70]:


fd2_2['Seasonal First Difference']=fd2_2['unemp_rate']-fd2_2['unemp_rate'].shift(3)


# In[71]:


adfuller_test(fd2_2['Seasonal First Difference'].dropna())


# In[72]:


adfuller_test(fd2_2['unemp_rate'].diff().diff().dropna())


# In[73]:


diff=fd2_2.unemp_rate.diff().dropna()

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
ax1.plot(diff)
ax1.set_title("after differencing")
plot_acf(diff, ax=ax2);


# In[74]:


plot_pacf(diff, ax=ax2)


# In[75]:


diff


# In[76]:


model = pm.auto_arima(y,start_p=1, max_p=3, d=None, max_d=1,start_q=1, max_q=3,  
                      start_P=1,max_P=3, D=None, max_D=1, start_Q=1, max_Q=3,
                      max_order=10, m=4, seasonal=True, information_criterion='aic',
                      test='adf',trace=True,random_state=10)


# In[77]:


model


# In[78]:


model.summary()


# In[79]:


model.plot_diagnostics(figsize=(10,8))
plt.show()


# In[80]:


pip install notebook --upgrade


# In[81]:


from statsmodels.tsa.arima.model import ARIMA

model=ARIMA(train_data, order = (0,1,1))
result_ARIMA = model.fit()
print(result_ARIMA.summary())


# In[82]:


model_fit=model.fit()


# In[83]:


model_fit.summary()


# In[84]:


pip install nest_asyncio==1.5.3


# In[85]:


import pandas as pd


# In[ ]:





# In[86]:


#forecasting

fc, conf=result_ARIMA.forecast(20, alpha=0.05)

fc=pd.Series(fc,index=test_data.index)
lower=pd.Series(conf[:,0], index=test_data.index)
upper=pd.Series(conf[:,1], index=test_data.index)

plt.figure(figsize=(10,8), dpi=100)
plt.plot(train_data, label='training data')
plt.plot(test_data, color = 'blue', label='Actual')
plt.plot(fc, color = 'orange',label='Predicted')
plt.fill_between(lower.index, lower, upper, color='k', alpha=.10)
plt.xlabel('date')
plt.ylabel('unemp_rate')
#plt.plot(filepath['date'],macro_data['unemp_rate'])
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:


model = sm.tsa.statespace.SARIMAX(endog= result_ARIMA(), trend='c', order=(0,1,0))
fitted = model.fit(disp=False)
print(fitted.summary())

result = fitted.forecast(57, alpha =0.05)

# Make as pandas series
fc_series = pd.Series(result[564:620],test.index)
lower_series = pd.Series(result[564], test.index)
upper_series = pd.Series(result[620], test.index)

# Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(df_log, label='training data')
plt.plot(test, color = 'blue', label='Actual data')
plt.plot(fc_series, color = 'orange',label='Predicted data')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='gray', alpha=.10)
plt.xlabel('date')
plt.ylabel('unemp_rate')
plt.legend(loc='best', fontsize=8)
plt.show()


# In[ ]:


fc


# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
model_sa = SARIMAX(y, order=(0,1,1),seasonal_order=(1,0,0,4),enforce_stationarity=False,
enforce_invertibility=False)
results=model_sa.fit()
model_sa_fit = model_sa.fit(disp=0)


# In[ ]:


results.summary()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(test_data, fc))
print('Mean Squared Error:', metrics.mean_squared_error(test_data, fc))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_data, fc)))
print('Accuracy', metrics.r2_score(test_data, fc))


# In[ ]:


import statsmodels.api as sm


# In[ ]:


model=sm.tsa.statespace.SARIMAX(fd2_2['unemp_rate'],order=(0,1,1),seasonal_order=(0,1,1,4))
results=model.fit()
results.summary()


# In[ ]:


fd2_2['date'].dt.to_period('Q')


# In[ ]:


fd2_2 = fd2_2.set_index(pd.DatetimeIndex(fd2_2['date']))


# In[ ]:


from pandas.tseries.offsets import DateOffset
future_dates=[fd2_2.index[-1]+ DateOffset(months=x) for x in range(0,24)]


# In[ ]:


future_datest_df=pd.DataFrame(index=future_dates[1:],columns=fd2_2.columns)


# In[ ]:


future_datest_df.tail()


# In[ ]:


future_df=pd.concat([fd2_2,future_datest_df])


# In[ ]:


future_df['forecast'] = results.predict(start = 88, end = 96, dynamic= True)  
future_df[['unemp_rate', 'forecast']].plot(figsize=(12, 8))
future_df


# In[ ]:


from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm_notebook
from itertools import product


# In[ ]:


type(df.date[0])


# In[ ]:


filepath = 'new_dataset.csv'
macro_data = pd.read_csv(filepath, parse_dates=True, index_col='date')
print(macro_data.shape) 
macro_data.head()


# In[ ]:


macro_data['deposits_rate'] = macro_data['deposits_rate'].astype(float)


# In[ ]:


macro_data['int_rate'] = macro_data['int_rate'].astype(float)


# In[ ]:


macro_data.head()


# In[ ]:


macro_data.columns


# In[ ]:


#macro_data.set_index('date', inplace=True)


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = macro_data[macro_data.columns[i]]
    ax.plot(data, linewidth=1)
    # Decorations
    ax.set_title(macro_data.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# In[ ]:


ad_fuller_result_1 = adfuller(macro_data['int_rate'].diff()[1:])

print('int_rate')
print(f'ADF Statistic: {ad_fuller_result_1[0]}')
print(f'p-value: {ad_fuller_result_1[1]}')

print('\n---------------------\n')

ad_fuller_result_2 = adfuller(macro_data['deposits_rate'].diff()[1:])

print('deposits_rate')
print(f'ADF Statistic: {ad_fuller_result_2[0]}')
print(f'p-value: {ad_fuller_result_2[1]}')


# In[ ]:


print('int_rate causes deposits_rate?\n')
print('------------------')
granger_1 = grangercausalitytests(macro_data[['deposits_rate', 'int_rate']], 7)


print('\n\deposits_rate causes int_rate?\n')
print('------------------')
granger_2 = grangercausalitytests(macro_data[['int_rate', 'deposits_rate']], 7)


# In[ ]:


macro_data = macro_data[['deposits_rate','int_rate']]
macro_data


# In[ ]:


train_df=macro_data[:-24]
test_df=macro_data[-24:]
print(test_df)


# In[ ]:


print(test_df.shape)


# In[ ]:


model = VAR(train_df.diff()[7:])


# In[ ]:


sorted_order=model.select_order(maxlags=20)
print(sorted_order.summary())


# In[ ]:


var_model = VARMAX(train_df, order=(19,0),enforce_stationarity= True)
fitted_model = var_model.fit(disp=False)
print(fitted_model.summary())


# In[ ]:


len(train_df.index)


# In[ ]:


n_forecast = 24
predict = fitted_model.get_prediction(start=len(train_df.index),end=len(train_df.index) + n_forecast-1)
predictions=predict.predicted_mean
predictions


# In[ ]:


predictions.columns=['int_rate_predicted','deposits_rate_predicted']
predictions


# In[ ]:


test_vs_pred=pd.concat([test_df,predictions],axis=1)


# In[ ]:


test_vs_pred.plot(figsize=(12,5))


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(test_df, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(test_df, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_df, predictions)))


# 

# In[ ]:


model_arch_1_1 = arch_model(df['int_rate'][1:], mean = "constant", vol = "ARCH", p = 1)
results_arch_1_1 = model_arch_1_1.fit(update_freq = 5)
results_arch_1_1.summary()


# In[ ]:


import ARCH

# a standard GARCH(1,1) model
garch = arch_model(returns, vol='garch', p=1, o=0, q=1)
garch_fitted = garch.fit()

# one-step out-of sample forecast
garch_forecast = garch_fitted.forecast(horizon=1)
predicted_et = garch_forecast.mean['h.1'].iloc[-1]


# In[ ]:


pip install arch 


# In[ ]:


pip install arch_model


# In[ ]:


import arch
import pandas as pd
from arch import arch_model

model_arch_1_1 = arch_model(df['int_rate'][1:], mean = "constant", vol = "ARCH", p = 1)
results_arch_1_1 = model_arch_1_1.fit(update_freq = 5)
results_arch_1_1.summary()


# In[ ]:


model_garch_1_1 = arch_model(df.int_rate[1:], mean = "constant", vol = "GARCH", p = 1, q = 1)
results_garch_1_1 = model_garch_1_1.fit(update_freq = 5)
results_garch_1_1.summary()


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X, y)

regr.predict([[0, 0, 0, 0]])

regr.score(X, y)


# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)

print(regr.predict([[0, 0, 0, 0]]))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

features = pd.read_csv('new_dataset.csv')
features.head(5)


# In[ ]:


print('The shape of our features is:', features.shape)


# In[ ]:


features.describe()


# In[ ]:


features = pd.get_dummies(features)
features.iloc[:,5:].head(5)


# In[ ]:


# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(features['actual'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# In[ ]:


import numpy as np

rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng)

regr_1.fit(X, y)
regr_2.fit(X, y)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("colorblind")

plt.figure()
plt.scatter(X, y, color=colors[0], label="training samples")
plt.plot(X, y_1, color=colors[1], label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, color=colors[2], label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("unemp_rate")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()


# In[ ]:




