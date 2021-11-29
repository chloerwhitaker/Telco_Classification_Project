 Telco Classification Project 

    About the Project 

    Project Goals
        My goal is to identify the drivers of churn at Telco and the likelihood that a current customer will churn, so that we can develop custom solutions that increase retention. 
     
    Project Description
        A huge expense for telecommunication companies is aquiring new customers. If telecommunication companies like ours can retain current customers, they can save on the cost of aquisition. 
        In this project we will look at factors that contribute to customers churnning or not churnning. By identifiying the drivers of churn, we will then develop a model to predict how likely a future customer is to churn or not. We will also recommend tailored solutions to keep customers happy and with the company. Keeping customers = saving money, and we can help telecommunication companies do both. 
    
    Initial Questions
        1. What month are customers most likely to churn in?    
        2. Are customers with a certain service type more or less likely to churn? 
             - Specifically are customers with fiber more likely to churn? 
        3. Do the customers that churn have a higher monthly cost than those that do not churn? 
            - Do the customers that churn have a higher monthly cost and a certain service type? 
        4. Do the customers that churn use paperless billing more than those who don't?

    Data Dictionary 

        Here is a list of every variable used in this project along with its meaning. 

        | Variable | Meaning |
| --- | --- |
| senior_citizen | 'Yes': 1, 'No': 0 if they are a senior citizen |
| partner | 'Yes': 1, 'No': 0 if they have a partner on their contract|
| dependents| 'Yes': 1, 'No': 0 if they have dependents on their contract|
| tenure | the number of months a customer has been with the company|
| phone_servive | 'Yes': 1, 'No': 0 if they have phone service|
| multiple_lines | 'Yes': 1, 'No': 0 if they have multiple lines|
| online_security | 'Yes': 1, 'No': 0 if they have online security|
|online_backup|'Yes': 1, 'No': 0 if they have online backup |
|device_protection|'Yes': 1, 'No': 0 if they have device protection |
|tech_support| 'Yes': 1, 'No': 0 if they have tech support|
|streaming_tv| 'Yes': 1, 'No': 0 if they have tv streaming|
|streaming_movies|'Yes': 1, 'No': 0 if they have movie streaming|
|paperless_billing|'Yes': 1, 'No': 0 if they have paperless billing |
|monthly_charges| customer's monthly bill| 
|total_charges  |total of monthly bills during customer's tenure |
|churn|'Yes': 1, 'No': 0 if they have churned |
|contract_type| one year, two year, or month-to-month|
|internet_service_type | dsl, fiber optic, or no service|
|payment_type | mailed check, electronic check, credit card (automatic), bank transfer (automatic) |
|is_autopay| yes if payment_type contains 'automatic' |
|contract_type_month_to_month| yes or no |
|contract_type_one_year|yes or no  |
|contract_type_two_year|yes or no  |
|internet_service_type_DSL|yes or no  |
|internet_service_type_fiber_optic |yes or no  |
|internet_service_type_None|yes or no  |
|payment_type_bank_transfer_auto|yes or no  |
|payment_type_credit_card_auto| yes or no |
|payment_type_electronic_check|yes or no  | 
|payment_type_mailed_check| yes or no | 


    Imports Used: 

    - import acquire_telco
    - import prepare_telco

    - import warnings
    - warnings.filterwarnings("ignore")

    - import numpy as np
    - import pandas as pd 
    - import math

    - from pydataset import data

    - import scipy.stats as stats

    - from sklearn.model_selection import train_test_split
    - from sklearn.tree import DecisionTreeClassifier
    - from sklearn.tree import export_graphviz
    - from sklearn.metrics import classification_report
    - from sklearn.metrics import confusion_matrix
    - from sklearn.ensemble import RandomForestClassifier
    - from sklearn.neighbors import KNeighborsClassifier
    - from sklearn.impute import SimpleImputer
    - from sklearn.linear_model import LogisticRegression

    - import matplotlib.pyplot as plt
i   - mport seaborn as sns

     Steps to Reproduce 

     1. you will need to import everything listed in imports used
     2. you will need a env file with your username, password, and host giving you access to Codeup's SQL server (make sure you also have a .gitignore in your github repo to protect your env file!)
     3. clone this repo containing the Telco_Classification_Report as well as my prepare_telco.py, and acquire_telco.py
     4. that should be all you need to do run the Telco_Classification_Report!

     The Plan 
        - Create a final report 
        - Create a working notebook 
        - Acquire
            - Create acuire_telco.py
            - Add to github
            - Aquire telco data in working notebook
        - Prepare 
            - Create prepare_telco.py 
            - Add to github
            - Prepare telco data in working notebook
        - Explore (most of time spent here)
            - Add 4 best asked/answered questions to report 
                    - 2 analyzed through visuals/statistical tests
                    - 2 just visuals
        - Develop models
            - Add 3 best models to report
                - show fitting model
                - show evaluacting model
                - show selecting model
            - On best model 
                - show visually how is preformed against test data
        - Refine report
            - show only necessary/important steps/data
            - add intro/Goals
            - add conclusion/takeaways/next steps
            - comment all code
            - markdown all processes
    
    Takeaways
        _ We start losing customers drastically after the fisrt month, and those that churn are most likely to leave in the 18th month.

        - Customers paying for fiber optic internet service are more likely to churn than those who aren't.

        - The reason for this and churn in general seems to be that those who leave have a higher monthly bill than those who stay.

        - The customers that churn that use paperless billing, still pay more per month than those who stay.

    Summary

        - In this project we found a correlation between customers with a higher monthly bill and those that churn. 

        - Fiber optic internet is one of the main places Telco is hemeraging customers and once again, it seems to correlate to cost. 

        - We have also developed a logistic regression model that can predict whether a future customer will churn or not with 81% accuracy. 

    Recommendations 

        - In addition to the effort to lower fiber costs through bundling or other means, it would also be a smart solution to create an easy to use portal where customers can see their monthly bill and add and drop features to tailor their services to their needs and budget. Customers could alos use this portal to automatically their bill. 
        
        - Although our current model is beating baseline predictions by 8%, with more time we can adjust the features and hyperpameters to tune the model  to better predict future churn. 