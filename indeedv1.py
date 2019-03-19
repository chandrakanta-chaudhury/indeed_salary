#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:51:25 2019

@author: chandrakantachaudhury
"""
from __future__ import print_function
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
import pandas as pd
import numpy as np
import os


h2o.init()



os.chdir("/home/chandrakantachaudhury/Desktop/HACKTHON/indeed/indeed_data_science_exercise")

#paste here

from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features=6, input_type='string')

##Prediction on new data

testdf = pd.read_csv("test_features_2013-03-07.csv") 
testdf.columns

#perform preprocessing of test data features similar to the train data 

#feature enginerring for companyId (hash function as it has 63 categories so it will cause a huge sparse matrix)
hashed_features = fh.fit_transform(testdf['companyId'])
hashed_features = hashed_features.toarray()
testdf=pd.concat([testdf, pd.DataFrame(hashed_features)], 
          axis=1)

testdf.rename(columns={0: 'comp1', 1: 'comp2',2: 'comp3',3: 'comp4',4: 'comp5',5: 'comp6'}, inplace=True) 

#feature engineering for other categorical features (one hot encoding)

testdf = pd.get_dummies(testdf, columns = ['jobType','degree','major','industry'])


#scaling the continous features

from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()

robustdf=testdf.iloc[:,2:4]
robust_scaled_df = robust_scaler.fit_transform(robustdf)
robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['yearsExperience','milesFromMetropolis'])
testdf=testdf.drop(['yearsExperience','milesFromMetropolis'],axis=1)

testdf=pd.concat([testdf,robust_scaled_df],axis=1) 

##arranging columns  as per it was trained 
testdf=testdf.drop(['companyId'],axis=1)


cols=['jobId','comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6', 'yearsExperience',
                   'milesFromMetropolis', 'jobType_CEO', 'jobType_CFO',
                   'jobType_CTO', 'jobType_JANITOR', 'jobType_JUNIOR', 'jobType_MANAGER',
                   'jobType_SENIOR', 'jobType_VICE_PRESIDENT', 'degree_BACHELORS',
                   'degree_DOCTORAL', 'degree_HIGH_SCHOOL', 'degree_MASTERS',
                   'degree_NONE', 'major_BIOLOGY', 'major_BUSINESS', 'major_CHEMISTRY',
                   'major_COMPSCI', 'major_ENGINEERING', 'major_LITERATURE', 'major_MATH',
                   'major_NONE', 'major_PHYSICS', 'industry_AUTO', 'industry_EDUCATION',
                   'industry_FINANCE', 'industry_HEALTH', 'industry_OIL',
                   'industry_SERVICE', 'industry_WEB']
testdf_format=testdf[cols].copy()

testdf_format=testdf.drop(['jobId'],axis=1)

#predict for test data provided 
#converting to H2O frame before prediction 
h2oframe = h2o.H2OFrame(testdf_format)

#load saved model

model_path="/home/chandrakantachaudhury/Desktop/HACKTHON/indeed/indeed_data_science_exercise/indeedv01"
saved_model = h2o.load_model(model_path)

#prediction
predicted=saved_model.predict(h2oframe)

#converting into pandas dataframe
prediction= h2o.as_list(predicted)

prediction["jobId"]=testdf["jobId"]

predict=prediction.filter(["jobId","predict"]).copy()
predict.rename(columns={'predict':'salary'},inplace=True)

#dumping to csv 
 
predict.to_csv("test_salaries.csv",index=False)

import psutil
process = psutil.Process(os.getpid())
print(os.getpid())
print(process.memory_info().rss)


 
 


 
 
