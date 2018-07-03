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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix
from scipy import sparse
import pandas as pd
import re
from random import randrange, uniform
from scipy.stats import truncnorm
import surprise
from surprise import Dataset
from surprise import Reader
import pandas as pd 
from scipy.spatial import distance
from surprise.model_selection import cross_validate
import random
from surprise.model_selection import GridSearchCV


 
#dataset = pd.read_csv("Tweets.csv")
dataset = pd.read_csv("finalizedfull.csv")
#data = dataset['text']
data = dataset['tweet']
data1=data.reshape(len(dataset),1)

vectorizer = TfidfVectorizer(max_features=2000, min_df=10, stop_words='english')
X = vectorizer.fit_transform(data)
idx_to_word = np.array(vectorizer.get_feature_names())
print('kdsljf',X)
# apply NMF
nmf = NMF(n_components=20, init='nndsvd', random_state=0, max_iter = 200)
np_topics1=20
#nmf = NMF(n_components=np_topics1)
W=[]
W = nmf.fit_transform(X)
tweets_topics=W
S=W
sum_rows=np.full(len(W),0.0)
sum_rows=W.sum(axis=1,dtype=float)
'''for i in range(len(W)):
 for j in range(len(W[i])):
   #tweets_topics[i][j]=round(W[i,j]/sum_rows[i],3)
   tweets_topics[i][j]=W[i,j]/sum_rows[i]
data_temp=data[:10]
'''
tweets_topics=np.full((100,np_topics1),0.0)   
for i in range(100):
 for j in range(len(W[0])):
   tweets_topics[i][j]=(W[i,j]/sum_rows[i])
print(tweets_topics)
H = nmf.components_
#=============================End Topic Modelling==================================

#===================================Ground Truth====================================
#dataset = pd.read_csv("Tweets.csv")
dataset = pd.read_csv("finalizedfull.csv")
#documents = dataset[['airline_sentiment','text']]
documents = dataset[['tweet','senti']]

documents.replace({0: 1, 4: 2, 2: 3}, inplace=True)
#groundTruth=documents['airline_sentiment']
groundTruth=documents['senti']
print(groundTruth)
groundTruth_temp=groundTruth[:100]
np_tweets=len(tweets_topics)
np_topics=len(tweets_topics[0])
nb_annotators=10
nb_labels=3
annt_responses=np.full((nb_annotators,np_tweets),0)
annt_topics=np.full((nb_annotators,np_topics),1.0)
trueLabels=[]
topics=np.zeros(np_topics)
for i in range(0,np_topics):
   for j in range(0,np_tweets):
        topics[i]=topics[i]+tweets_topics[j,i]
normalizingFactor=topics.sum()
#========================================================================

def simulateLabels(groundTruth,tweets_topics,topics,numberAnnotators,meanAccuarcy,SdAccuracy,meanLikelihood):
  x=[]
  likelihood=[]
  numberTopics=len(tweets_topics[0])
  numberTweets=len(tweets_topics)
    
  for m in range(0,numberAnnotators):
   done=False
   while(done==False):
    val=np.random.normal(meanAccuarcy, SdAccuracy,1)#the accurcy for each annotater
    if val>0 and val<1:
        x.append(val)
        done=True

  for m in range(0,numberAnnotators):
   done=False
   while(done==False):
    #val=np.random.normal(meanLikelihood,SdLikelihood,1)#the Likelihood of response for each annotater
    val=np.random.exponential(meanLikelihood,1)
    if val>0.0 and val<1.0:
       likelihood.append(val)
       done=True
       
  for m in range(0,numberAnnotators):
   done=False
   while(done==False):
    if x[m]>0.0 and x[m]<1.0:
     for i in range(0,numberTweets):
        correct=np.random.binomial(1,x[m],1)
        annotate=np.random.binomial(1,likelihood[m],1)
        if (annotate[0]!=0.0):
         if correct[0]==1:   
          annt_responses[m,i]=groundTruth[i]
          for c in range(0,numberTopics):
                if tweets_topics[i,c]!=0.0:
                    #annt_topics[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]/topics[c]),4)
                    annt_topics[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]),4)
                    
         else:
           annt_responses[m,i]=randrange(1,nb_labels+1,1)  
           if annt_responses[m,i]==groundTruth[i]:
               for c in range(0,np_topics):
                if tweets_topics[i,c]!=0.0:
                    #annt_topics[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]/topics[c]),4)
                    annt_topics[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]),4)
           else:
               for c in range(0,numberTopics):
                if tweets_topics[i,c]!=0.0:
                    #annt_topics[m,c]=round(annt_topics[m,c]-(tweets_topics[i,c]/topics[c]),4)
                    annt_topics[m,c]=round(annt_topics[m,c]-(tweets_topics[i,c]),4)
             
    done=True
  return (annt_topics,annt_responses)


#===========generate labels===============================       
def fillMissingValues(annt_responses):
        annotator_responses=[]
        annotator_responses=annt_responses
        for i in range(0,nb_annotators):
            
            trainingSet=[]
           
            trainingTarget=[]
            test=[]
            accuracy=0
            f=0
            for j in range(0,np_tweets):
                 if annotator_responses[i][j]!= 0:
                    trainingSet.append(tweets_topics[j].reshape(-1))
                    trainingTarget.append(annotator_responses[i][j])
            text_clf= MultinomialNB()
            trainingSetArray=np.asarray(trainingSet)
            if(len(trainingSetArray)!=0):
                print(trainingSetArray)
                text_clf = text_clf.fit( np.asarray(trainingSet),trainingTarget)
                for j in range(0,np_tweets):
                     if annotator_responses[i][j]== 0:
                       test=[]
                       test.append(tweets_topics[j])
                       predicted = text_clf.predict(test)
                       annotator_responses[i][j]=predicted[0]
                       if annt_responses[i][j]==groundTruth[j]:
                            accuracy=accuracy+1
                       f=f+1 
        return annotator_responses
#==========================inter-agreement===========================================

def inter_agreement(annotators_responses,numberTopics):
  nb_annotators=len(annotators_responses)
  np_tweets=len(annotators_responses[0])
  agree=[]
  for i in range(0,nb_annotators):
    agreement=0.0
    for j in range(0,nb_annotators):
      for z in range(0,np_tweets):
        if annt_responses[i][z]!=0:
         if annt_responses[i][z]==annt_responses[j][z] and i!=j:
           agreement=agreement+1
    agree.append(agreement/(np_tweets))
  return agree
#===============End inter-agreement==================================
#=============== Kappa inter-agreement=============================
def kappa_aggreement(annt_responses,nb_labels):
    kappa_agree=[]
    nb_annotators=len(annt_responses)
    np_tweets=len(annt_responses[0])
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
        pra=0.0
        if total!=0.0:
         pra=np.trace(confusion)/total
        pre=0.0
        cols=confusion.sum(axis=0)
        rows=confusion.sum(axis=1)
        for i in range(0,nb_labels):
          if total!=0.0:
            pre=pre+(cols[i]*rows[i])/total
        if total!=0.0: 
         pre=pre/total
        kappa=kappa+((pra-pre)/(1.0-pre))
     kappa_agree.append(kappa/(norm))
    return kappa_agree
 
#===============End Kappa inter- agreement========================
#=====================inter agreement with topics
#
def interAgreementWithTopics(annt_responses,np_topics,nb_labels):
    agree=inter_agreement(annt_responses,np_topics)
    trueLabels=[]
    np_tweets=len(annt_responses[0])
    nb_annotators=len(annt_responses)
    annt_tpc=np.full((len(annt_responses),np_topics),1.0)
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
    return (trueLabels,annt_tpc) 
#====================End inter agreement with topics=================================================
#============Kappa with topics============================
def kappaInteragreemtWithTopics(annt_responses,nb_labels,tweets_topics):
    kappa_agree=kappa_aggreement(annt_responses,nb_labels)
    nb_annotators=len(annt_responses)
    np_tweets=len(annt_responses[0])
    np_topics=len(tweets_topics[0])
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
    return(Kappa_trueLabels,annt_tpc_kappa)
#==============End Kappa with topics=================================
#=======================inter agreement without topics===============================
def interAgreementWithoutTopics(annt_responses,nb_labels):
    trueLabelsWithoutTopics=[]
    agree=inter_agreement(annt_responses,np_topics)
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
    return trueLabelsWithoutTopics
#======================End inter agreement without topics
def reliability_interAgreement(trueLabels,tweets_topics,annt_responses,topics,nb_labels):
    nb_annotators=len(annt_responses)
    np_topics=len(tweets_topics[0])
    annt_tpc2=np.full((nb_annotators,np_topics),1.0)
    trueLabels=interAgreementWithoutTopics(annt_responses,nb_labels)
    agree=inter_agreement(annt_responses,np_topics)
    for j in range(0,len(annt_responses)):
      for i in range(0,len(annt_responses[0])):
          if (annt_responses[j][i]==trueLabels[i]) :
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
    return(trueLabels_agree_rel)
trueLabels_agree_rel=reliability_interAgreement(trueLabels,tweets_topics,annt_responses,topics,nb_labels)
#====================End intr agreement then reliability=========================
#============Kappa without topics============================
def kappaInterAgreementWithoutTopics(annt_responses,nb_labels):
    Kappa_trueLabelsWithoutTopics=[]
    kappa_agree=kappa_aggreement(annt_responses,nb_labels)
    np_tweets=len(annt_responses[0])
    nb_annotators=len(annt_responses)
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
    return Kappa_trueLabelsWithoutTopics
#==============End Kappa without topics=================================
#====================Kappa interagreement then reliability==================
def reliability_kappaInterAgreement(tweets_topics,annt_responses,nb_labels):
    np_topics=len(tweets_topics[0])
    Kappa_annt_tpc2=np.full((nb_annotators,np_topics),1.0)
    kappa_agree=kappa_aggreement(annt_responses,nb_labels)
    np_tweets=len(annt_responses[0])
    Kappa_trueLabelsWithoutTopics=kappaInterAgreementWithoutTopics(annt_responses,nb_labels)
    for j in range(0,nb_annotators):
     for i in range(0,np_tweets):
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
                   sim=sim+(kappa_agree[j]*Kappa_annt_tpc2[j,c])
         if highsim<sim:
             truelabel=label
             highsim=sim
     kappa_trueLabels_agree_rel.append(truelabel)
    return kappa_trueLabels_agree_rel
#====================End Kappa intr agreement then reliability=========================
#===========Majority Voting===============================
def majorityVoting(annt_responses,nb_labels):
    majority_voting=[]
    np_tweets=len(annt_responses[0])
    nb_annotators=len(annt_responses)
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
    return majority_voting

def reliability_majority_voting(majority_voting,tweets_topics,annt_responses,topics,nb_labels):
    nb_annotators=len(annt_responses)
    np_topics=len(tweets_topics[0])
    annt_tpc2=np.full((nb_annotators,np_topics),1.0)
    trueLabels=majority_voting
    for j in range(0,len(annt_responses)):
      for i in range(0,len(annt_responses[0])):
          if (annt_responses[j][i]==trueLabels[i]) :
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
                   sim=sim+annt_tpc2[j,c]
         if highsim<sim:
             truelabel=label
             highsim=sim
     trueLabels_agree_rel.append(truelabel)
    return(trueLabels_agree_rel)

def mvWithTopics(annt_responses,np_topics,nb_labels):
    np_tweets=len(annt_responses[0])
    trueLabels=[]
    nb_annotators=len(annt_responses)
    annt_tpc=np.full((len(annt_responses),np_topics),1.0)
    for i in range(0,np_tweets):
     highsim=0.0
     truelabel=0
     for label in range(1,nb_labels+1):
         sim=0.0
         for j in range(0,nb_annotators):
             if annt_responses[j][i]==label:
                 for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0:
                   sim=sim+(annt_tpc[j,c])
         if highsim<sim:
             truelabel=label
             highsim=sim
     trueLabels.append(truelabel)
     for j in range(0,nb_annotators):
      if (annt_responses[j][i]==trueLabels[i]) :
              for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0:
                    #annt_tpc[j,c]=round(annt_tpc[j,c]+(tweets_topics[i,c]/topics[c]),4)
                    annt_tpc[j,c]=round(annt_tpc[j,c]+(tweets_topics[i,c]),4)
      else:
             for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0:
                    #annt_tpc[j,c]=round(annt_tpc[j,c]-(tweets_topics[i,c]/topics[c]),4)
                    annt_tpc[j,c]=round(annt_tpc[j,c]-(tweets_topics[i,c]),4)
    return (trueLabels,annt_tpc) 
def accuracy(trueLabels,groundTruth,text):
    hits=0
    np_tweets=len(trueLabels)
    for i in range(0,np_tweets):
     if groundTruth[i]==trueLabels[i]:
        hits=hits+1
    print('accuracy',text,(float(hits)/float(np_tweets)),'missclasification: ',np_tweets-hits,'of',np_tweets)
def compareReliability(realAnntTopics,estAnntTopics):
        nb_annotators=len(realAnntTopics)
        np_topics=len(realAnntTopics[0])
        unbiased_annt_topics=np.full((nb_annotators,np_topics),' ')
        unbiased_annt_tpcs=np.full((nb_annotators,np_topics),' ')
        for i in range(0,nb_annotators):
            for j in range(0,np_topics):
                if realAnntTopics[i][j]-realAnntTopics.mean()>0.0:
                     unbiased_annt_topics[i][j]='r'
                else:
                     unbiased_annt_topics[i][j]='u'
                if estAnntTopics[i][j]-estAnntTopics.mean()>0.0:
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
        '''print('Real Reliability of annotators')
        print(unbiased_annt_topics)
        print('Estimated Reliability of annotators')
        print(unbiased_annt_tpcs)
        '''
        counter=0
        for i in range(0,nb_annotators):
            for j in range(0,np_topics):
                if unbiased_annt_tpcs[i][j]!=unbiased_annt_topics[i][j]:
                    counter=counter+1
        print('accuarcy of annotators reliability',counter/(nb_annotators*np_topics))
#========================accuracy======================================

print('without the missing values')
meanAccuarcy=0.7
SdAccuracy=0.1
meanLikelihood=1.0
(annt_topics,annt_responses)=simulateLabels(groundTruth,tweets_topics,topics,nb_annotators,meanAccuarcy,SdAccuracy,meanLikelihood)
agree=inter_agreement(annt_responses,np_topics)
(trueLabels,annt_tpc)=interAgreementWithTopics(annt_responses,np_topics,nb_labels)
(Kappa_trueLabels,annt_tpc_kappa)=kappaInteragreemtWithTopics(annt_responses,nb_labels,tweets_topics)
trueLabelsWithoutTopics=interAgreementWithoutTopics(annt_responses,nb_labels)
Kappa_trueLabelsWithoutTopics=kappaInterAgreementWithoutTopics(annt_responses,nb_labels)
kappa_trueLabels_agree_rel= reliability_kappaInterAgreement(tweets_topics,annt_responses,nb_labels)
majority_voting=majorityVoting(annt_responses,nb_labels)
rel_MV=reliability_majority_voting(majority_voting,tweets_topics,annt_responses,topics,nb_labels)
(mv_withTopics,mv_annt_tpc)=mvWithTopics(annt_responses,np_topics,nb_labels) 
accuracy(trueLabels,groundTruth_temp,'inter-agreement with topics')
compareReliability(annt_topics,annt_tpc)
accuracy(trueLabelsWithoutTopics,groundTruth_temp,'inter-agreement without topics')
accuracy(trueLabels_agree_rel,groundTruth_temp,'inter-agreement then reliability')
accuracy(Kappa_trueLabelsWithoutTopics,groundTruth_temp,'Kappa without Topics')
accuracy(Kappa_trueLabels,groundTruth_temp,'Kappa Topics')
compareReliability(annt_topics,annt_tpc_kappa)
accuracy(kappa_trueLabels_agree_rel,groundTruth_temp,'Kappa then reliability')
accuracy(majority_voting,groundTruth_temp,'Majority Voting')
accuracy(mv_withTopics,groundTruth_temp,'Majority Voting with Topics')
compareReliability(annt_topics,mv_annt_tpc)
accuracy(rel_MV,groundTruth_temp,'Majority Voting then reliability')

annotator_responses=fillMissingValues(annt_responses)
print('with the missing values')

agree=inter_agreement(annotator_responses,np_topics)
(trueLabels,annt_tpc)=interAgreementWithTopics(annotator_responses,np_topics,nb_labels)
(Kappa_trueLabels,annt_tpc_kappa)=kappaInteragreemtWithTopics(annotator_responses,nb_labels,tweets_topics)
trueLabelsWithoutTopics=interAgreementWithoutTopics(annotator_responses,nb_labels)
Kappa_trueLabelsWithoutTopics=kappaInterAgreementWithoutTopics(annotator_responses,nb_labels)
kappa_trueLabels_agree_rel= reliability_kappaInterAgreement(tweets_topics,annotator_responses,nb_labels)
majority_voting=majorityVoting(annotator_responses,nb_labels)
rel_MV=reliability_majority_voting(majority_voting,tweets_topics,annotator_responses,topics,nb_labels)
(mv_withTopics,mv_annt_tpc)=mvWithTopics(annotator_responses,np_topics,nb_labels) 
accuracy(trueLabels,groundTruth_temp,'inter-agreement with topics')
accuracy(trueLabelsWithoutTopics,groundTruth_temp,'inter-agreement without topics')
accuracy(trueLabels_agree_rel,groundTruth_temp,'inter-agreement then reliability')
accuracy(Kappa_trueLabelsWithoutTopics,groundTruth_temp,'Kappa without Topics')
accuracy(Kappa_trueLabels,groundTruth_temp,'Kappa Topics')
accuracy(kappa_trueLabels_agree_rel,groundTruth_temp,'Kappa then reliability')
accuracy(majority_voting,groundTruth_temp,'Majority Voting')
accuracy(mv_withTopics,groundTruth_temp,'Majority Voting with Topics')
accuracy(rel_MV,groundTruth_temp,'Majority Voting then reliability')


#===================End Majority Voting===ُ================================

print('number of tweets',np_tweets)
print('number of topics',np_topics)
print('number of annotators',nb_annotators)
#======================================================================
count=0
for i in range(len(annt_responses)):
    for j in range(len(annt_responses[i])):
         if annt_responses[i][j]==0:
             count=count+1
            
print ('sparsity of the responses matrix',count/(np_tweets*nb_annotators),'empty:',count,'of ',np_tweets*nb_annotators)