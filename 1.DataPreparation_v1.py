
# coding: utf-8

# In[262]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95%}</style>"))


# In[202]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[7]:


import pandas as pd
pd.set_option('max_columns', None)


# In[118]:


#df = pd.read_excel('/tcad2/client16/TataWire_Thick_MTB/1.OriginalData/PQM_DATA_Thick MTB.xlsx', sheetname='WorkingSheet')
df = df.rename(columns={'Drawing Speed, m/s':'Drawing Speed', 'SR Temp oC':'SR Temp'})
df.head()


# In[ ]:





# In[17]:


df_desc = df.describe()
df_desc


# In[26]:


df.head()


# In[29]:


df['Rolling Date'].min()


# In[31]:


df['Rolling Date'].max()


# In[52]:


df_describe_o = df.describe(include=['O'])
df_describe_o


# In[92]:


data1 = [df.shape[0]]*len(df_describe_o.columns)
col_list = list(df_describe_o.columns)


# In[77]:


pd.DataFrame(data=[data1],columns=col_list,index=['total']).append(df.describe(include=['O']))


# In[81]:


def describe_total_o(df):
    total = [df.shape[0]]*len(df_describe_o.columns)
    col_list = list(df_describe_o.columns)
    desc = pd.DataFrame(data=[data1],columns=col_list,index=['total']).append(df.describe(include=['O']))
    return desc


# In[82]:


describe_total_o(df)


# In[84]:


df.shape


# In[93]:


df[df.duplicated(['Cast No','WRM Coil No.'], keep=False)][['Cast No','WRM Coil No.']].sort_values(['Cast No','WRM Coil No.']).head(15)


# ### Duplicates in finish coil no. 

# In[101]:


t1 = df['Finish coil no'].value_counts().reset_index()
dup_finish_coil_nos = t1[t1['Finish coil no'] > 1]['index'].tolist()
df[df['Finish coil no'].isin(dup_finish_coil_nos)]


# ## Data Preparation

# In[102]:


df.head()


# ### Defining Column Lists

# In[182]:


identifyer_col = ['Rolling Date','Drawing Date','Outlier','WRM No', 'Cast No','WRM Coil No.','Finish coil no','Machine No. ', 'Master Spool No.','Coil No. after Die Set change']

feature_list = ['N', 'C', 'Mn', 'Si','Cr', 'S', 'P', 'Ni', 'Cu', 'CE', 'LHT', 'Blower1', 'Blower2','Blower3','Blower4', 'Blower5', 'Blower6', 
                'UTS Min', 'UTS Max', 'UTS Avg', 'RA', 'Ageing',  'SR Temp', 'Drawing Speed',
                 'BL Samp 1', 'BL Samp 2', 'Tor Samp 1', 'Tor Samp 2', 'Tor Samp 3']

target_var = ['Final Remark']


# In[242]:


df_clean = df[identifyer_col + feature_list + target_var]


# In[243]:


df_clean.dtypes


# ## Data Cleansing

# In[244]:


## cleaning of LHT(Laying Head Temperature) and SR_Temp(Stress Relieving Temperature)

import re
def get_range_avg(txt):
    if pd.notnull(txt) and re.search('(\d.*)-(\d.*)', txt):
        low_bound = re.search('(\d.*)-(\d.*)', txt).group(1)
        upper_bound = re.search('(\d.*)-(\d.*)', txt).group(2)
        return (int(low_bound) + int(upper_bound))/2

    
def convert_float(txt):
    if pd.notnull(txt):
        if str(txt).isdigit():
            return float(txt)
        elif re.search('(\d.*)/(\d.*)', txt):
            return float(re.search('(\d.*)/(\d.*)', txt).group(2))
        

df_clean['LHT'] = df_clean['LHT'].apply(get_range_avg)  
df_clean['SR Temp'] = df_clean['SR Temp'].apply(get_range_avg) 

df_clean['BL Samp 1'] = df_clean['BL Samp 1'].apply(convert_float)
df_clean['Tor Samp 2'] = df_clean['Tor Samp 2'].apply(convert_float)



# In[245]:


df_clean.dtypes


# In[246]:


df_clean.describe()


# In[247]:


df_clean['Final Remark'].value_counts()


# In[271]:


# Invalid ageging found - Removing invalid
df_clean[df_clean['Ageing'] < 0]


# ### Output the Result

# In[272]:


from time import gmtime, strftime
yyyymmdd = strftime("%Y%m%d", gmtime())
df_clean.to_csv('/tcad2/client16/TataWire_Thick_MTB/2.PreparedData/thick_mtb_'+ yyyymmdd +'.csv', index=False)
df_clean.shape


# ---
# ## Rough from here

# In[263]:





# In[ ]:




