# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:26:57 2018

@author: HP
"""

#================================Topic Modelling============================================

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.decomposition import NMF
from array import array

 
dataset = pd.read_csv("Tweets.csv")
data = dataset['text']
vectorizer = TfidfVectorizer(max_features=2000, min_df=20, stop_words='english')
X = vectorizer.fit_transform(data)
idx_to_word = np.array(vectorizer.get_feature_names())
# apply NMF
nmf = NMF(n_components=20, init='random', tol=1, random_state=0, max_iter = 200)
W = nmf.fit_transform(X)
tweets_topics=W
S=W
sum_rows=W.sum(axis=1)

for i in range(len(W)):
 for j in range(len(W[i])):
   tweets_topics[i][j]=round(W[i,j]/sum_rows[i],3)
data_temp=data[:10]


tweets_topics=np.zeros((10,20))   
for i in range(10):
 for j in range(len(W[i])):
   tweets_topics[i][j]=round(W[i,j]/sum_rows[i],3)

H = nmf.components_
#=============================End Topic Modelling==================================

#===================================Ground Truth====================================
import pandas as pd
import re
dataset = pd.read_csv("Tweets.csv")
documents = dataset[['airline_sentiment','text']]
documents.replace({'neutral': 1, 'positive': 2, 'negative': 3}, inplace=True)
groundTruth=documents['airline_sentiment']
groundTruth_temp=groundTruth[:10]
#================================End Ground Truth===================================

#==============================Simulate labels===========================================
from random import randrange, uniform
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import surprise
from surprise import NMF
from surprise import Dataset
from surprise import Reader
import pandas as pd 
from scipy.spatial import distance
from surprise.model_selection import cross_validate
import random
from surprise.model_selection import GridSearchCV

np_tweets=len(tweets_topics)
np_topics=len(tweets_topics[0])
nb_annotators=10
nb_labels=3
x=[]
likelihood=[]
annt_responses=np.full((nb_annotators,np_tweets),0)
annt_topics=np.full((nb_annotators,np_topics),1.0)
#===========generate labels===============================       
topics=[]
topics=np.zeros(np_topics)

for i in range(0,np_topics):
    for j in range(0,np_tweets):
        topics[i]=topics[i]+tweets_topics[j,i]
    
for m in range(0,nb_annotators):
 done=False
 while(done==False):
   val=np.random.normal(0.6, 0.2,1)#the accurcy for each annotater
   if val>0 and val<1:
       x.append(val)
       done=True

for m in range(0,nb_annotators):
 done=False
 while(done==False):
   val=np.random.normal(0.5, 0.1,1)#the Likelihood of response for each annotater
   if val>0.0 and val<1.0:
       likelihood.append(val)
       done=True
for m in range(0,nb_annotators):
  done=False
  while(done==False):
   if x[m]>0.0 and x[m]<1.0:
    for i in range(0,np_tweets):
        correct=np.random.binomial(1,x[m],1)
        annotate=np.random.binomial(1,likelihood[m],1)
        if (annotate[0]!=0.0):
         if correct[0]==1:   
          annt_responses[m,i]=groundTruth[i]
          for c in range(0,np_topics):
                if tweets_topics[i,c]!=0.0:
                    annt_topics[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]/topics[c]),4)
         else:
           annt_responses[m,i]=randrange(1,nb_labels+1,1)  
           if annt_responses[m,i]==groundTruth[i]:
               for c in range(0,np_topics):
                if tweets_topics[i,c]!=0.0:
                    annt_topics[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]/topics[c]),4)
           else:
               for c in range(0,np_topics):
                if tweets_topics[i,c]!=0.0:
                    annt_topics[m,c]=round(annt_topics[m,c]-(tweets_topics[i,c]/topics[c]),4)
             
   done=True
#=========================End generate label==========================
#==============================End simulate labels=========================================
#====================Annotators as Classifiers==============================================
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
 }

for i in range(0,nb_annotators):
    trainingSet=[]
    trainingTarget=[]
    test=[]
    for j in range(0,np_tweets):
         if annt_responses[i][j]!= 0:
            trainingSet.append(data[j])
            trainingTarget.append(groundTruth[j])
    text_clf = Pipeline([('vect', TfidfVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])
    text_clf = text_clf.fit(trainingSet, trainingTarget)
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(trainingSet, trainingTarget)
    gs_clf.best_score_
    gs_clf.best_params_
    for j in range(0,np_tweets):
         if annt_responses[i][j]== 0:
           test=[]
           test.append(data[j])
           predicted = text_clf.predict(test)
           annt_responses[i][j]=predicted[0]
print('complete',annt_responses)
#====================End annotators as classifiers=======================================   
#==========================inter-agreement===========================================
annt_tpc=np.full((nb_annotators,np_topics),1.0)
agree=[]
totalsim=0
trueLabels=[]
for i in range(0,nb_annotators):
 agreement=0
 for j in range(0,nb_annotators):
    for z in range(0,np_tweets):
      if annt_responses[i][z]!=0:
       if annt_responses[i][z]==annt_responses[j][z] and i!=j:
         agreement=agreement+1
 agree.append(agreement/(nb_annotators-1))
 totalsim=totalsim+agreement/(nb_annotators-1)
#===============End inter-agreement==================================
#=============== Kappa inter-agreement=============================
kappa_agree=[]
for i in range(0,nb_annotators):
 norm=0
 for j in range(0,nb_annotators):
    kappa=0.0
    common=False
    confusion=np.full((nb_labels,nb_labels),0)
    for z in range(0,np_tweets):
        for l in range(1,nb_labels+1):
           if annt_responses[i][z]!=0 and annt_responses[j][z]!=0:
               common=True 
               confusion[(annt_responses[i][z])-1][(annt_responses[j][z])-1]=confusion[(annt_responses[i][z])-1][(annt_responses[j][z])-1]+1
    if common==True:
        norm=norm+1
    total=confusion.sum()
    pra=np.trace(confusion)/total
    pre=0.0
    cols=confusion.sum(axis=0)
    rows=confusion.sum(axis=1)
    for i in range(0,nb_labels):
      pre=pre+(cols[i]*rows[i])/total
    pre=pre/total
    kappa=kappa+((pra-pre)/(1-pre))
 kappa_agree.append(kappa/(norm))
 
#===============End Kappa inter- agreement========================
#=====================inter agreement with topics

for i in range(0,np_tweets):
 highsim=0.0
 truelabel=0
 for label in range(1,nb_labels+1):
     sim=0.0
     for j in range(0,nb_annotators):
         if annt_responses[j][i]==label:
             for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
               sim=sim+(agree[j]*annt_tpc[j,c])
     if highsim<sim:
         truelabel=label
         highsim=sim
 trueLabels.append(truelabel)
 for j in range(0,nb_annotators):
  if (annt_responses[j][i]==trueLabels[i]) :
          for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
                annt_tpc[j,c]=round(annt_tpc[j,c]+(tweets_topics[i,c]/topics[c]),4)
  else:
         for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
                annt_tpc[j,c]=round(annt_tpc[j,c]-(tweets_topics[i,c]/topics[c]),4)
#====================End inter agreement with topics=================================================

#============Kappa with topics============================
annt_tpc_kappa=np.full((nb_annotators,np_topics),1.0)
Kappa_trueLabels=[]

for i in range(0,np_tweets):
 highsim=0.0
 truelabel=0
 for label in range(1,nb_labels+1):
     sim=0.0
     for j in range(0,nb_annotators):
         if annt_responses[j][i]==label:
             for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
               sim=sim+(kappa_agree[j]*annt_tpc_kappa[j,c])
     if highsim<sim:
         truelabel=label
         highsim=sim
 Kappa_trueLabels.append(truelabel)
 for j in range(0,nb_annotators):
  if (annt_responses[j][i]==trueLabels[i]) :
          for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
                annt_tpc_kappa[j,c]=round(annt_tpc_kappa[j,c]+(tweets_topics[i,c]/topics[c]),4)
  else:
         for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
                annt_tpc_kappa[j,c]=round(annt_tpc_kappa[j,c]-(tweets_topics[i,c]/topics[c]),4)
    
#==============End Kappa with topics=================================
#=======================inter agreement without topics===============================
trueLabelsWithoutTopics=[]
for i in range(0,np_tweets):
 highsim=0.0
 truelabel=0
 for label in range(1,nb_labels+1):
     sim=0.0
     for j in range(0,nb_annotators):
         if annt_responses[j][i]==label:
             for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
               sim=sim+agree[j]
     if highsim<sim:
         truelabel=label
         highsim=sim
 trueLabelsWithoutTopics.append(truelabel)

#======================End inter agreement without topics
#============Kappa without topics============================
Kappa_trueLabelsWithoutTopics=[]

for i in range(0,np_tweets):
 highsim=0.0
 truelabel=0
 for label in range(1,nb_labels+1):
     sim=0.0
     for j in range(0,nb_annotators):
         if annt_responses[j][i]==label:
             for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
               sim=sim+(kappa_agree[j])
     if highsim<sim:
         truelabel=label
         highsim=sim
 Kappa_trueLabelsWithoutTopics.append(truelabel)
    
#==============End Kappa without topics=================================

#====================interagreement then reliability==================
annt_tpc2=np.full((nb_annotators,np_topics),1.0)
for j in range(0,nb_annotators):
  if (annt_responses[j][i]==trueLabelsWithoutTopics[i]) :
          for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
                annt_tpc2[j,c]=round(annt_tpc2[j,c]+(tweets_topics[i,c]/topics[c]),4)
  else:
         for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
                annt_tpc2[j,c]=round(annt_tpc2[j,c]-(tweets_topics[i,c]/topics[c]),4)

trueLabels_agree_rel=[]
 
for i in range(0,np_tweets):
 highsim=0.0
 truelabel=0
 for label in range(1,nb_labels+1):
     sim=0.0
     for j in range(0,nb_annotators):
         if annt_responses[j][i]==label:
             for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
               sim=sim+(agree[j]*annt_tpc2[j,c])
     if highsim<sim:
         truelabel=label
         highsim=sim
 trueLabels_agree_rel.append(truelabel)
 
#====================End intr agreement then reliability=========================

#====================Kappa interagreement then reliability==================
Kappa_annt_tpc2=np.full((nb_annotators,np_topics),1.0)
for j in range(0,nb_annotators):
  if (annt_responses[j][i]==Kappa_trueLabelsWithoutTopics[i]) :
          for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
                Kappa_annt_tpc2[j,c]=round(Kappa_annt_tpc2[j,c]+(tweets_topics[i,c]/topics[c]),4)
  else:
         for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
                Kappa_annt_tpc2[j,c]=round(Kappa_annt_tpc2[j,c]-(tweets_topics[i,c]/topics[c]),4)

kappa_trueLabels_agree_rel=[]
 
for i in range(0,np_tweets):
 highsim=0.0
 truelabel=0
 for label in range(1,nb_labels+1):
     sim=0.0
     for j in range(0,nb_annotators):
         if annt_responses[j][i]==label:
             for c in range(0,np_topics):
              if tweets_topics[i,c]!=0.0:
               sim=sim+(agree[j]*Kappa_annt_tpc2[j,c])
     if highsim<sim:
         truelabel=label
         highsim=sim
 kappa_trueLabels_agree_rel.append(truelabel)
 
#====================End Kappa intr agreement then reliability=========================




#===========Majority Voting===============================
majority_voting=[]
for j in range(0,np_tweets):
        high=0
        s=0
        for x in range(1,nb_labels+1):
         s=0
         for i in range(0,nb_annotators):
            if annt_responses[i][j]==x:
                s=s+1
         if s>high:
          high=s
          majority=x
        majority_voting.append(majority)  
#===================End Majority Voting===ُ================================
unbiased_annt_topics=np.full((nb_annotators,np_topics),' ')
unbiased_annt_tpcs=np.full((nb_annotators,np_topics),' ')
for i in range(0,nb_annotators):
    for j in range(0,np_topics):
        if annt_topics[i][j]-annt_topics.mean()>0.0:
             unbiased_annt_topics[i][j]='r'
        else:
             unbiased_annt_topics[i][j]='u'
        if annt_tpc[i][j]-annt_tpc.mean()>0.0:
             unbiased_annt_tpcs[i][j]='r'
        else:
             unbiased_annt_tpcs[i][j]='u'
for i in range(0,nb_annotators):
   spammer=True
   for j in range(0,np_topics):       
       if unbiased_annt_topics[i][j]=='r':
        spammer=False
   if spammer==True:
       for c in range(0,np_topics):
           unbiased_annt_topics[i][c]='s'
           
for i in range(0,nb_annotators):
   spammer=True
   for j in range(0,np_topics):       
       if unbiased_annt_tpcs[i][j]=='r':
        spammer=False
   if spammer==True:
       for c in range(0,np_topics):
           unbiased_annt_tpcs[i][c]='s'
print('Real Reliability of annotators')
print(unbiased_annt_topics)
print('Estimated Reliability of annotators')
print(unbiased_annt_tpcs)
counter=0
for i in range(0,nb_annotators):
    for j in range(0,np_topics):
        if unbiased_annt_tpcs[i][j]!=unbiased_annt_topics[i][j]:
            counter=counter+1
#========================accuracy======================================
hits_algo1=0
hits_MV=0
hits_NMF=0
hits_withoutTopics=0
hits_Agre_rel=0
hits_kappa=0
hits_kappaWithoutTopics=0
kappa_hits_Agre_rel=0

for i in range(0,np_tweets):
    if groundTruth[i]==trueLabels[i]:
        hits_algo1=hits_algo1+1
    if majority_voting[i]==groundTruth[i]:
        hits_MV=hits_MV+1
#    if  trueLabelsNMF[i]==groundTruth[i]:
#         hits_NMF=hits_NMF+1
    if  trueLabelsWithoutTopics[i]==groundTruth[i]:
          hits_withoutTopics=hits_withoutTopics+1
    if  trueLabels_agree_rel[i]==groundTruth[i]:
          hits_Agre_rel=hits_Agre_rel+1
    if  Kappa_trueLabels[i]==groundTruth[i]:
          hits_kappa=hits_kappa+1
    if  Kappa_trueLabelsWithoutTopics[i]==groundTruth[i]:
          hits_kappaWithoutTopics=hits_kappaWithoutTopics+1
    if  kappa_trueLabels_agree_rel[i]==groundTruth[i]:
          kappa_hits_Agre_rel=kappa_hits_Agre_rel+1      
          
print('number of tweets',np_tweets)
print('number of topics',np_topics)
print('number of annotators',nb_annotators)
print('accuracy of inter-agreement with topics',hits_algo1/np_tweets,'missclasification: ',np_tweets-hits_algo1,'of',np_tweets)
print('accuracy of Kappa inter-agreement topics',hits_kappa/np_tweets,'missclasification:',np_tweets-hits_kappa,'of',np_tweets) 

print('accuracy of inter-agreement without topics',hits_withoutTopics/np_tweets,'missclasification:',np_tweets-hits_withoutTopics,'of',np_tweets) 
print('accuracy of Kappa inter-agreement without topics',hits_kappaWithoutTopics/np_tweets,'missclasification:',np_tweets-hits_kappaWithoutTopics,'of',np_tweets) 

print('accuracy of inter-agreement then reliability',hits_Agre_rel/np_tweets,'missclasification: ',np_tweets-hits_Agre_rel,'of',np_tweets)
print('accuracy of kappa inter-agreement then reliability',kappa_hits_Agre_rel/np_tweets,'missclasification: ',np_tweets-kappa_hits_Agre_rel,'of',np_tweets)

print('accuracy of Majority Voting',hits_MV/np_tweets,'missclasification:',np_tweets-hits_MV,'of',np_tweets) 
print('accuracy of annotators reliability',((np_topics*nb_annotators)-counter)/(np_topics*nb_annotators),'missclasification:',counter,'of',(np_topics*nb_annotators)) 

#print('accuracy of NMF',hits_NMF/np_tweets,np_tweets-hits_NMF)        
#======================================================================
count=0
for i in range(len(annt_responses)):
    for j in range(len(annt_responses[i])):
         if annt_responses[i][j]==0:
             count=count+1
            
print ('sparsity of the responses matrix',count/(np_tweets*nb_annotators),'empty:',count,'of ',np_tweets*nb_annotators)