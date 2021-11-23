#!/usr/bin/env python
# coding: utf-8

# In[1]:


import acquire_telco

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


np.random.seed(123)


# In[2]:


df = acquire_telco.new_telco_data()
df.head()


# In[4]:


df.dtypes


# In[18]:


def wrangle_telco(df):
    """
    This function queries the new_telco_data and 
    drops columns and whitespaces, 
    converts to correct datatype, 
    convert binary categorical variables to numeric,
    gets dummies from non-binary object varibales,
    concatenates dummy dataframe to original,
    drops the object columns we created dummies from,
    and returns a nice and clean telco df
    """
    df = acquire_telco.new_telco_data()
    
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id', 'gender'], inplace=True)

    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype (object to float)
    df['total_charges'] = df.total_charges.astype(float)
    
    # Convert binary categorical variables to numeric
    df['tenure'] = df.tenure.replace(0, 1)
    df['churn'] = df.churn.replace({'Yes': 1, 'No': 0})
    df['partner'] = df.partner.replace({'Yes': 1, 'No': 0})
    df['dependents'] = df.dependents.replace({'Yes': 1, 'No': 0})
    df['paperless_billing'] = df.paperless_billing.replace({'Yes': 1, 'No': 0})
    df['phone_service'] = df.phone_service.replace({'Yes': 1, 'No': 0})
    df['multiple_lines'] = df.multiple_lines.replace({'Yes': 1, 'No': 0, 'No phone service': 0})
    df['online_security'] = df.online_security.replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['streaming_movies'] = df.streaming_movies.replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['streaming_tv'] = df.streaming_tv.replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['online_backup'] = df.online_backup.replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['device_protection'] = df.device_protection.replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['tech_support'] = df.tech_support.replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['is_autopay'] = df.payment_type.str.contains('automatic')
    
    # Get dummies from non-binary object varibales 
    dummies_df = pd.get_dummies(df[['contract_type', 'internet_service_type', 'payment_type']], dummy_na=False)
    
    # Concatenate dummy dataframe to original 
    df_with_dummies = pd.concat([df, dummies_df], axis=1)
    
    # Drop the object columns we created dummies from 
    df = df_with_dummies.drop(columns=['contract_type', 'internet_service_type', 'payment_type'])
   
    return df


# In[6]:


telco_cleaned = wrangle_telco(df)
telco_cleaned.head()


# In[7]:


telco_cleaned.dtypes


# In[19]:


def train_validate_test_split(telco_cleaned, seed=123):
    '''
    This function takes in a dataframe (df) and returns 3 dfs
    (train, validate, and test) split 20%, 24%, 56% respectively. 
    
    Also takes in a random seed for replicating results.  
    '''
    train_and_validate, test = train_test_split(
        telco_cleaned, test_size=0.2, random_state=seed, stratify=telco_cleaned.churn
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.churn,
    )
    return train, validate, test


# In[12]:


train, validate, test = train_validate_test_split(telco_cleaned)


# In[13]:


train.head()


# In[14]:


train.dtypes


# In[20]:


def clean_split_titanic_data(df):
    ''' 
    This function runs both the wrangle_telco and train_validate_test_split functions.
    It takes in the original df and returns the split dfs train, validate, test (in that order).
    '''
    telco_cleaned = wrangle_telco(df)
    train, validate, test = train_validate_test_split(telco_cleaned , seed=123)
    return train, validate, test


# In[16]:


train, validate, test = clean_split_titanic_data(df)


# In[17]:


train.head()


# In[ ]:




