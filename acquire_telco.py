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


# In[ ]:




