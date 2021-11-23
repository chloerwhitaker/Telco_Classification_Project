#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
from env import host, user, password


# In[6]:


def get_connection(db, user=user, host=host, password=password):
    '''
    This function establishes a connection to the Codeup database
    using the credentials in my env file.
    The argument required should be the database name entered as a string.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[4]:


def new_telco_data():
    '''
    This function reads the telco data from the Codeup database into a 
    dataframe.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Reads in the DataFrame from the Codeup database.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df


# In[32]:


#def get_telco_data():
#    '''
#    This function checks to see if the telco data is already stored
#    locally as a csv. 
#    If not, it reads in the Telco data from the Codeup database
#    and writes it to a csv file
#    If it exists, it reads the existing file.
#    '''
#    if os.path.isfile('telco_wrangled.csv'):
#        
#        # Reads this file if it already exists.
#        df = pd.read_csv('telco_wrangled.csv', index_col=0)
#        
#    else:
#        
#        # If csv not already local, fresh data is read into a DataFrame
#        df = wrangle_telco()
#        
#        # Cache data
#        df.to_csv('telco_wrangled.csv')
#        
#    return df


# In[ ]:




