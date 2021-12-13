#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 23:30:52 2021

@author: prateekwadhwani
"""

import os
cwd = os.getcwd()
print(cwd)

import pandas as pd

data_entertainment_train = pd.read_csv('Data/Triage Data Jaccard/jaccardEntertainmentNews_Similarity_Dataset.csv')
data_politics_train = pd.read_csv('./Data/Triage Data Jaccard/jaccardPoliticsNews_Similarity_Dataset.csv')
data_sports_train = pd.read_csv('./Data/Triage Data Jaccard/jaccardSportsNews_Similarity_Dataset.csv')
data_tech_train = pd.read_csv('./Data/Triage Data Jaccard/jaccardTechNews_Similarity_Dataset.csv')
data_travel_train = pd.read_csv('./Data/Triage Data Jaccard/jaccardTravelNews_Similarity_Dataset.csv')
data_finance_train = pd.read_csv('./Data/Triage Data Jaccard/jaccardFinanceNews_Similarity_Dataset.csv')

data_all_train=pd.concat([data_entertainment_train,data_sports_train,data_tech_train,data_travel_train,data_politics_train,data_finance_train],axis = 0)
  
data_all_train.to_csv('./Data/Triage Data Jaccard/jaccardNews_Similarity_Dataset.csv',index = False)


data_entertainment_train = pd.read_csv('Data/Triage Data TfIdf/tfidfEntertainmentNews_Similarity_Dataset.csv')
data_politics_train = pd.read_csv('./Data/Triage Data TfIdf/tfidfPoliticsNews_Similarity_Dataset.csv')
data_sports_train = pd.read_csv('./Data/Triage Data TfIdf/tfidfSportsNews_Similarity_Dataset.csv')
data_tech_train = pd.read_csv('./Data/Triage Data TfIdf/tfidfTechNews_Similarity_Dataset.csv')
data_travel_train = pd.read_csv('./Data/Triage Data TfIdf/tfidfTravelNews_Similarity_Dataset.csv')
data_finance_train = pd.read_csv('./Data/Triage Data TfIdf/tfidfFinanceNews_Similarity_Dataset.csv')

data_all_train=pd.concat([data_entertainment_train,data_sports_train,data_tech_train,data_travel_train,data_politics_train,data_finance_train],axis = 0)
  
data_all_train.to_csv('./Data/Triage Data TfIdf/tfidNews_Similarity_Dataset.csv',index = False)


data_entertainment_train = pd.read_csv('Data/Triage Data Glove/gloveEntertainmentNews_Similarity_Dataset.csv')
data_politics_train = pd.read_csv('./Data/Triage Data Glove/glovePoliticsNews_Similarity_Dataset.csv')
data_sports_train = pd.read_csv('./Data/Triage Data Glove/gloveSportsNews_Similarity_Dataset.csv')
data_tech_train = pd.read_csv('./Data/Triage Data Glove/gloveTechNews_Similarity_Dataset.csv')
data_travel_train = pd.read_csv('./Data/Triage Data Glove/gloveTravelNews_Similarity_Dataset.csv')
data_finance_train = pd.read_csv('./Data/Triage Data Glove/gloveFinanceNews_Similarity_Dataset.csv')

data_all_train=pd.concat([data_entertainment_train,data_sports_train,data_tech_train,data_travel_train,data_politics_train,data_finance_train],axis = 0)
  
data_all_train.to_csv('./Data/Triage Data Glove/gloveNews_Similarity_Dataset.csv',index = False)






# data_tech_train = pd.read_csv('./data/Triage Data Glove/gloveTechNews_Similarity_Dataset.csv')
# data_tech_train['Class'] = 'Entertainment'
# data_tech_train.to_csv('Data/Triage Data Glove/gloveEntertainmentNews_Similarity_Dataset.csv')


# cates = ['Entertainment','Sports','Politics','Travel','Tech','Finance']
# for cate in cates:
#     data_train = pd.read_csv('Data/Triage Data Jaccard/jaccard'+cate+'News_Similarity_Dataset.csv')
#     data_train['Class'] = cate
#     data_train.to_csv('Data/Triage Data Jaccard/jaccard'+cate+'News_Similarity_Dataset.csv')