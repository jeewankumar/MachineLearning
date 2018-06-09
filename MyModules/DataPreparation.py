from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = 12,7.5
plt.rcParams['figure.dpi'] = 100
plt.style.use('ggplot')

def say_hello():
	print("hello!")

## Clean columns given in a dataset
import re
def cleanColumns(col_list):
	col_list1 = []
	for col in col_list:
		col = ' '.join(word[0].upper() + word[1:] for word in col.split())
		col = re.sub('\W', '', col)
		col_list1.append(col)
	return col_list1
	
## Give descrition of all object column with total	
def describe_total_o(df):
    total = [df.shape[0]]*len(df_describe_o.columns)
    col_list = list(df_describe_o.columns)
    desc = pd.DataFrame(data=[data1],columns=col_list,index=['total']).append(df.describe(include=['O']))
    return desc