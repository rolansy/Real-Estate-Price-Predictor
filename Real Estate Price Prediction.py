#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize']=(20,10)


# In[3]:


df=pd.read_csv('Bengaluru_House_Data.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.groupby('area_type')['area_type'].count()


# In[7]:


df1=df.drop(['area_type','availability','society','balcony'],axis=1)


# In[8]:


df1


# In[9]:


df1.isnull().sum()


# In[10]:


df2=df1.dropna()
df2.isnull().sum()


# In[11]:


df2.shape


# In[12]:


df2['size'].unique()


# In[13]:


df2['bhk']=df2['size'].apply(lambda x: int(x.split(' ')[0]))


# In[14]:


df2['bhk'].unique()


# In[15]:


df2[df2.bhk>20]


# In[16]:


df2.total_sqft.unique()


# In[17]:


def is_float(x):
    try: 
        float(x)
    except:
        return False
    return True


# In[18]:


df2[~df2['total_sqft'].apply(is_float)].head(10)


# In[19]:


def convert_sqft_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return  ((float(tokens[0])+float(tokens[1]))/2)
    try :
        return float(x)
    except:
        return None


# In[20]:


convert_sqft_to_num('32223 - 81231')


# In[21]:


df3=df2.copy()
df3.total_sqft=df3['total_sqft'].apply(convert_sqft_to_num)


# In[22]:


df3.total_sqft.unique()


# In[23]:


df3.total_sqft.isnull().sum()


# In[24]:


df3.loc[30]


# In[25]:


df5=df3.copy()
df5 [ 'price_per_sqft'	]=df5[ 'price' ]*100000/df5[ 'total_sqft' ]
df5


# In[26]:


len(df5.location.unique())


# In[27]:


df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5.groupby('location')['location'].count()
location_stats


# In[28]:


len(location_stats[location_stats<=10])


# In[29]:


location_stats_less_than_10=location_stats[location_stats<=10]
location_stats_less_than_10


# In[30]:


df5.location=df5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[31]:


df5


# In[32]:


df5.groupby('location')['location'].count()


# In[33]:


df5[df5.total_sqft/df5.bhk<300]


# In[34]:


df6=df5[~(df5.total_sqft/df5.bhk<300)]


# In[35]:


df6


# In[36]:


df6.price_per_sqft.describe()


# In[37]:


def remove_pps_outliers(df) :
    df_out=pd.DataFrame()
    for key, subdf in df.groupby( 'location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std( subdf.price_per_sqft )
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out =pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


# In[38]:


df7=remove_pps_outliers(df6)
df7


# In[39]:


df7.shape


# In[40]:


df7[(df7.location=='1st Block Jayanagar')&(df7.bhk==3)]


# In[41]:


def plot_loc_bhk_price(df,location):
    bhk2=df[(df.location==location)&(df.bhk==2)]
    bhk3=df[(df.location==location)&(df.bhk==3)]
    mpl.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 BHK',s=50)
    plt.xlabel('Total SQFT')
    plt.ylabel('Price per SQFT')
    plt.title(location)
    plt.legend()


# In[42]:


plot_loc_bhk_price(df7,'1st Block Jayanagar')


# In[43]:


plot_loc_bhk_price(df7,'Rajaji Nagar')


# In[44]:


plot_loc_bhk_price(df7,'Hebbal')


# In[45]:


def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location,loc_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in loc_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in loc_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


# In[46]:


df8=remove_bhk_outliers(df7)


# In[47]:


df8


# In[48]:


plot_loc_bhk_price(df8,'Hebbal')


# In[49]:


plot_loc_bhk_price(df8,'Rajaji Nagar')


# In[50]:


mpl.rcParams[ "figure.figsize" ]=(20, 10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel('Prce per sqft')
plt.ylabel('COunt')


# In[51]:


df8.bath.unique()


# In[52]:


df8.head(1)


# In[53]:


df8[df8.bhk+2<df8.bath]


# In[54]:


df9=df8[(df8.bhk+2>df8.bath)]
df9


# In[55]:


df10=df9.drop(['size','price_per_sqft'],axis=1)
df10


# In[ ]:





# In[56]:


pd.get_dummies(df10.location,dtype=int)


# In[57]:


dummies=pd.get_dummies(df10.location,dtype=int)
dummies


# In[58]:


df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')


# In[59]:


df11


# In[60]:


df12=df11.drop('location',axis=1)
df12


# In[61]:


df.shape,df12.shape


# In[62]:


x=df12.drop('price',axis='columns')


# In[63]:


y=df12.price


# In[64]:


from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.2,random_state=10)


# In[65]:


from sklearn.linear_model import LinearRegression as lr
model=lr()
model.fit(xtrain,ytrain)


# In[66]:


model.score(xtrain,ytrain),model.score(xtest,ytest)


# In[67]:


from sklearn.model_selection import ShuffleSplit


# In[68]:


from sklearn.model_selection import cross_val_score as cvs


# In[69]:


cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)


# In[70]:


cv


# In[71]:


cvs(lr(),x,y,cv=cv)


# In[72]:


from sklearn.model_selection import GridSearchCV as gscv 


# In[73]:


from sklearn.linear_model import Lasso


# In[74]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:





# In[75]:


def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'copy_X' : [True, False],
                'fit_intercept' : [True, False],
                'n_jobs' : [1,2,3],
                'positive' : [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  gscv(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

# find_best_model_using_gridsearchcv(X,y)


# In[86]:


x.columns


# In[92]:


x


# In[81]:


np.where(x.columns=='Vijayanagar')[0][0]


# In[85]:


np.zeros((2,3))


# In[93]:


def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(x.columns==location)[0][0]
    a=np.zeros(len(x.columns))
    a[0]=sqft
    a[1]=bath
    a[2]=bhk
    if loc_index>0:
        a[loc_index]=1
    return model.predict([a])[0]


# In[ ]:





# In[95]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[97]:


predict_price('1st Phase JP Nagar',1000,3,3)


# In[98]:


predict_price('1st Phase JP Nagar',1000,2,3)


# In[99]:


predict_price('1st Phase JP Nagar',1000,3,2)


# In[101]:


import pickle 
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(model,f)


# In[103]:


import json
columns={
    'data_columns':[col.lower() for col in x.columns]
}
with open ('columns.json','w') as f:
    f.write(json.dumps(columns))


# In[ ]:




