
# coding: utf-8

# # WoE and IV Calculations & Imputation

# In[140]:


from IPython.core.display import display, HTML
display(HTML("<style>.container {width:95%}</style>"))


# In[141]:


import pandas as pd
import numpy as np
import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')



# In[142]:


working_dir = '/mnt2/client16/20180501_ThickMTB'
iv_analyis_dir = '/mnt2/client16/20180501_ThickMTB/IV_Analysis/iv_result2/'


# ## Recalculating WoE and IV - Based on changes in the bins

# In[147]:


# Read the IV table
iv_table = pd.read_excel(working_dir+'/IV_Analysis/IV_Analysis.xlsx', sheetname='iv_table_club1')
iv_table.head()

## Caluculation of IV values based on c

iv_total = iv_table[iv_table.Cutpoint=='Total'][['CharName','CharBinNum','Cutpoint','CntRec','CntGood','CntBad']]
iv_total = iv_total.rename(columns={'CntRec':'TotalCntRec','CntGood':'TotalCntGood','CntBad':'TotalCntBad'})

iv_table1 = iv_table[iv_table.Cutpoint!='Total'][['CharName','CharBinNum','Cutpoint','CntRec','CntGood','CntBad']]
iv_table2 = iv_table1.groupby(['CharName','CharBinNum','Cutpoint']).sum().reset_index()


iv_table2 = pd.merge(iv_table2, iv_total.drop(['CharBinNum','Cutpoint'], axis=1), on='CharName',how='inner')
iv_table2['GoodRate'] = iv_table2['CntGood'] / iv_table2['TotalCntGood'] 
iv_table2['BadRate'] = iv_table2['CntBad'] / iv_table2['TotalCntBad']
iv_table2['WoE'] = np.log(iv_table2['GoodRate']/iv_table2['BadRate'])
iv_table2['IV'] = (iv_table2['GoodRate'] - iv_table2['BadRate'])*iv_table2['WoE']

iv_table2 = iv_table2[['CharName','CharBinNum','Cutpoint','CntRec','CntGood','CntBad','GoodRate','BadRate','WoE','IV']]
iv_table2 = iv_table2.replace([np.inf, -np.inf], np.nan)

iv_table2_total = iv_table2.groupby(['CharName']).sum().reset_index().drop(['CharBinNum','WoE'], axis=1)
iv_table2_total.merge(iv_total[['CharName','CharBinNum','Cutpoint']], on='CharName', how='inner')

## getting 'CharBinNum','Cutpoint' from original total
iv_table2_total = iv_table2_total.merge(iv_total[['CharName','CharBinNum','Cutpoint']], on='CharName', how='left')


iv_table_final = pd.concat([iv_table2,iv_table2_total])
iv_table_final = iv_table_final.merge(iv_table2_total[['CharName','IV']].rename(columns={'IV':'IV_Char'}), on='CharName', how='left')


iv_table_final = iv_table_final[['CharName','CharBinNum','Cutpoint','CntRec','CntGood','CntBad','GoodRate','BadRate','WoE','IV','IV_Char']]
iv_table_final = iv_table_final.sort_values(['CharName','CharBinNum']).reset_index()

iv_table_final.to_csv(iv_analyis_dir+'iv_table_final.csv')


# ## Generating WoE Charts

# In[ ]:





# In[148]:


iv_table = iv_table_final

iv_vars = iv_table.CharName.unique()

for var in iv_vars:
    print(var)
    woe_data = iv_table[iv_table.CharName==var][['CharName','Cutpoint','CntGood','CntBad','WoE','IV']]
    iv_value = woe_data[woe_data['Cutpoint'] == 'Total']['IV'].values[0].round(3)
    woe_data = woe_data[(woe_data['WoE'].notnull()) & (woe_data['Cutpoint']!='Total')]
    woe_data = woe_data[~woe_data['WoE'].isin(['#NAME?'])]
    woe_data['WoE'] = woe_data['WoE'].astype(float)
    CntGood = woe_data[['Cutpoint','CntGood']].rename(columns={'CntGood':'Count'})
    CntGood['Target'] = '1'
    CntBad = woe_data[['Cutpoint','CntBad']].rename(columns={'CntBad':'Count'})
    CntBad['Target'] = '0'
    CountTarget = pd.concat([CntGood,CntBad])

    ## Plotting the good and bad count
    f, ax = plt.subplots(2, sharex=True, figsize=(16,9))
    f.suptitle("Variable - "+ var + " | IV="+str(iv_value), fontsize=20)
    sns.barplot(x='Cutpoint', y='Count',hue='Target', data=CountTarget, ax=ax[0])
    sns.pointplot(x='Cutpoint', y='WoE', data=woe_data, ax=ax[1])
    
    plt.xticks(rotation=45)
    plt.savefig(iv_analyis_dir+var+'.png',dpi=100)
    plt.close()


# In[ ]:





# ## WOE Imputation

# In[149]:


## Reading the dataset
df = pd.read_csv('/mnt2/client16/20180501_ThickMTB/2.PreparedData/thick_mtb_20180507.csv')
df.columns = [x.replace(' ', '') for x in df.columns] ## Cleaning the columns names
df.head()


# In[151]:


f = open('woe_imputation_logic.py', 'w')
f.write('import re\n')
f.write('import pandas as pd\n')
f.write('import numpy as np\n')
f.write('\n')
f.write('def woe_imputation(df):\n')
f.write('\tfor idx in df.index:\n')
f.write('\t\tprint(idx)\n')

iv_table_final.head(10)
iv_vars = iv_table_final.CharName.unique()
var = 'Ageing'
for var in iv_vars:
    cond1 = 'df.loc[idx,' +'"'+var+'"'+ '].round(3) '
    cond2 = ' : df.loc[idx,' +'"'+var+'_WoE'+'"'+']= '

    var_bins = iv_table_final[iv_table_final.CharName==var][['Cutpoint','WoE']]
    var_bins = var_bins[~var_bins.Cutpoint.isin(['Missing','Total'])]
    
    for i in var_bins.index:
        if re.match(r'^=([^=].*)',var_bins.loc[i,'Cutpoint']):
            cutpoint = '<' + var_bins.loc[i,'Cutpoint'].replace("'",'')
        else:
            cutpoint = var_bins.loc[i,'Cutpoint'].replace("'",'')
            
        if i==min(var_bins.index):
            f.write('\t\tif '+ cond1 + cutpoint + cond2 + str(var_bins.loc[i,'WoE']).replace('nan','np.nan')+'\n')
        else:
            f.write('\t\telif '+ cond1 + cutpoint + cond2 + str(var_bins.loc[i,'WoE']).replace('nan','np.nan')+'\n')
    f.write('\n')

f.write('\treturn df')
f.write('\n')
f.close()


# In[ ]:





# In[152]:


from importlib import reload
import woe_imputation_logic as woe


# In[153]:


reload(woe)
df1 = woe.woe_imputation(df.fillna(-99999))


# In[154]:


from time import gmtime, strftime
yyyymmdd = strftime("%Y%m%d", gmtime())
df1.to_csv('/mnt2/client16/20180501_ThickMTB/2.PreparedData/thick_mtb_woe_imputed_'+ yyyymmdd +'.csv', index=False)
df1.shape


# In[ ]:





# ## Testing the data imputation

# In[155]:


woe_var = [x for x in df1.columns if 'WoE' in x]
len(woe_var)


# In[156]:


#df2 = df1.replace(-99999,np.nan)
woe_var = [x for x in df1.columns if 'WoE' in x]
woe_freq_list = list()
for var in woe_var:
    woe_freq = df1[var].value_counts(dropna=False).reset_index().rename(columns={'index':'WoE',var:'count'})
    woe_freq['CharName'] = var.split('_')[0]
    woe_freq_list.append(woe_freq)

woe_freq = pd.concat(woe_freq_list)
woe_freq['WoE'] = woe_freq['WoE'].astype(str)
woe_freq.head()


# In[125]:


iv_table_final_cut = iv_table_final[['CharName','Cutpoint','CntRec','WoE']]
iv_table_final_cut['WoE'] = iv_table_final_cut['WoE'].astype(str)

iv_table_final_cut = iv_table_final_cut.merge(woe_freq, on=['CharName','WoE'], how='left')
iv_table_final_cut.to_csv('DataImputation_test.csv')


# In[ ]:





# In[100]:


df1['S_Round'] = df1['S'].round(3)


# In[ ]:


np.round


# In[111]:


str(df1.loc[11994,'S'].round(3))
df1.loc[11994,'S'].round(3)


# In[139]:


df1['DrawingSpeed'].value_counts()


# In[101]:


## Checking indivial values
df1[['S','S_WoE','S_Round']].sample(15)

