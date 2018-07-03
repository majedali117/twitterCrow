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
from scipy.stats import skewnorm
import csv
from pathlib import Path

#=============================Export annotators================================================

def simulateLabels(groundTruth,tweets_topics,topics,numberAnnotators,meanAccuarcy,SdAccuracy,meanLikelihood,nb_labels):
  x=[]
  likelihood=[]
  numberTopics=len(tweets_topics[0])
  numberTweets=len(tweets_topics)
  print(numberTweets)
  annt_responses=np.full((numberAnnotators,numberTweets),0)
  annt_topics=np.full((numberAnnotators,np_topics),1.0)
  annt_topics_float=np.full((numberAnnotators,np_topics),1.0)
  for m in range(0,numberAnnotators):
   done=False
   while(done==False):
    val=np.random.normal(loc=meanAccuarcy,scale=SdAccuracy ,size=1)#the accurcy for each annotater
    val=val/100
    
    if val>0 and val<1:
        x.append(val)
        done=True
        
  for m in range(0,numberAnnotators):
   done=False
   while(done==False):
    val1=np.random.exponential(scale=meanLikelihood,size=1)
    val1=val1/100
    if val1>0.0 and val1<1.0:
       likelihood.append(val1)
       done=True
       
  annt_counter=[]
  for m in range(0,numberAnnotators):
      counter=0
      for i in range(0,numberTweets):
        correct=np.random.binomial(1,x[m],1)
        annotate=np.random.binomial(1,likelihood[m],1)
        if (annotate[0]!=0.0):
         counter=counter+1
         if correct[0]==1:   
          annt_responses[m,i]=groundTruth[i]
          for c in range(0,numberTopics):
                if tweets_topics[i,c]!=0.0:
                    annt_topics[m,c]=annt_topics[m,c]+1
                    annt_topics_float[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]),4)
                    
         else:
           annt_responses[m,i]=randrange(1,nb_labels+1,1)  
           if annt_responses[m,i]==groundTruth[i]:
               for c in range(0,np_topics):
                if tweets_topics[i,c]!=0.0:
                    annt_topics[m,c]=annt_topics[m,c]+1
                    annt_topics_float[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]),4)
           '''else:
               for c in range(0,numberTopics):
                if tweets_topics[i,c]!=0.0:
                    #annt_topics[m,c]=round(annt_topics[m,c]-(tweets_topics[i,c]/topics[c]),4)
                    annt_topics[m,c]=round(annt_topics[m,c]-(tweets_topics[i,c]),4)
             '''
      
  return (annt_topics,annt_topics_float,annt_responses)

#=============== Kappa inter-agreement=============================
def kappa_aggreement(annt_responses,nb_labels):
    kappa_agree=[]
    nb_annotators=len(annt_responses)
    np_tweets=len(annt_responses[0])
    
    for i in range(0,nb_annotators):
     norm=0
     kappa=0.0
     for j in range(0,nb_annotators):
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
        for d in range(0,nb_labels):
          if total!=0.0:
            pre=pre+(cols[d]*rows[d])/total
        if total!=0.0: 
         pre=pre/total
        if pre!=1: 
         kappa=kappa+((pra-pre)/(1.0-pre))
     if (norm!=0):    
      kappa_agree.append(kappa/(norm))
     else:
      kappa_agree.append(0.0)  
    ##print('kappa_Agree',kappa_agree)
    return kappa_agree
 
#===============End Kappa inter- agreement========================
#============Kappa with topics============================
def kappaInteragreemtWithTopics(annt_responses,nb_labels,tweets_topics,groundTruth):
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
      if (annt_responses[j][i]==truelabel) :
              for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0:
                    annt_tpc_kappa[j,c]=round(annt_tpc_kappa[j,c]+(tweets_topics[i,c]),4)
                    ##annt_tpc_kappa[j,c]=annt_tpc_kappa[j,c]+1
      else:
             for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0 and annt_responses[j][i]!=0 :
                    annt_tpc_kappa[j,c]=round(annt_tpc_kappa[j,c]-(tweets_topics[i,c]),4)
    ##print('kappTopics',Kappa_trueLabels)
    ##print(annt_tpc_kappa)'''
    return(Kappa_trueLabels,annt_tpc_kappa)
#==============End Kappa with topics=================================
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

#===========Majority Voting===============================
def majorityVoting(annt_responses,nb_labels,nb_topics):
    majority_voting=[]
    annt_mv_tpc=np.full((len(annt_responses),nb_topics),1.0)
    np_tweets=len(annt_responses[0])
    nb_annotators=len(annt_responses)
    for j in range(0,np_tweets):
            high=0
            s=0
            majority=0
            for x in range(1,nb_labels+1):
             s=0
             for i in range(0,nb_annotators):
                if annt_responses[i][j]==x:
                    s=s+1
             if s>high:
              high=s
              majority=x
            majority_voting.append(majority)
            for l in range(0,nb_annotators):
             for c in range(0,nb_topics):
                if (annt_responses[l][j]==majority) and (tweets_topics[j][c]!=0) :
                    annt_mv_tpc[l][c]=annt_mv_tpc[l][c]+1
    return (majority_voting,annt_mv_tpc)

def mvWithTopics(annt_responses,np_topics,nb_labels,groundTruth):
    np_tweets=len(annt_responses[0])
    trueLabels=[]
    nb_annotators=len(annt_responses)
    annt_tpc=np.full((len(annt_responses),np_topics),1.0)
    for i in range(0,np_tweets):
     highsim=0.0
     truelabel=0
     counter=0
     for nA in range(0,len(annt_responses)):
         if(annt_responses[nA,i]!=0):
          counter=counter+1
     for label in range(1,nb_labels+1):
         sim=0.0
         accm=0
         for j in range(0,nb_annotators):
             if annt_responses[j][i]==label:
                 for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0:
                   accm=accm+(annt_tpc[j,c])
         if counter!=0:
          sim=accm/counter
         
         if highsim<sim:
             truelabel=label
             highsim=sim
     trueLabels.append(truelabel)
     for k in range(0,nb_annotators):
      if (annt_responses[k][i]==trueLabels[i]) :
              for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0:
                    #annt_tpc[j,c]=round(annt_tpc[j,c]+(tweets_topics[i,c]/topics[c]),4)
                    ##annt_tpc[j,c]=round(annt_tpc[j,c]+(tweets_topics[i,c]),4)
                    annt_tpc[k,c]=annt_tpc[k,c]+1
      '''else:
             for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0 and annt_responses[j][i]!=0:
                    #annt_tpc[j,c]=round(annt_tpc[j,c]-(tweets_topics[i,c]/topics[c]),4)
                    ##annt_tpc[j,c]=round(annt_tpc[j,c]-(tweets_topics[i,c]),4)
                    annt_tpc[j,c]=annt_tpc[j,c]-1'''
    return (trueLabels,annt_tpc)
 
def accuracy(trueLabels,groundTruth,text,annt_responses,tweet_topic,error):
    hits=0
    np_tweets=len(trueLabels)
    for i in range(0,np_tweets):
     if groundTruth[i]==trueLabels[i]:
        hits=hits+1
    print('accuracy',text,(float(hits)/float(np_tweets)),'missclasification: ',np_tweets-hits,'of',np_tweets)
    error=error+((np_tweets-hits)*(np_tweets-hits))
    return (error)

def accuracyReliability(realAnntTopics,estAnntTopics):
        nb_annotators=len(realAnntTopics)
        np_topics=len(realAnntTopics[0])
        error_reliablility=0.0
        for i in range(0,nb_annotators):
            for j in range(0,np_topics):
                error_reliablility=error_reliablility+(realAnntTopics[i][j]-estAnntTopics[i][j])*(realAnntTopics[i][j]-estAnntTopics[i][j])
        error_reliablility=error_reliablility/(nb_annotators*np_topics)
        result=math.sqrt(error_reliablility)
        return result

#========================accuracy======================================
      
def mainRunWithSparsity(annt_responses,annt_topics,annt_topics_float,nb_labels,tweets_topics,groundTruth_temp,err,err_rel):
    np_topics=len(annt_topics[0])
    trueLabels=[]
    annt_tpc=[]
    majority_voting=[]
    annt_mv_tpc=[]
    mv_withTopics=[]
    mv_annt_tpc=[]
    print(annt_responses.shape)
    (trueLabels,annt_tpc)=kappaInteragreemtWithTopics(annt_responses,nb_labels,tweets_topics,groundTruth_temp)
    
    err_rel[0,0]=accuracyReliability(annt_topics_float,annt_tpc)
    err[0,0]=accuracy(trueLabels,groundTruth_temp,'Kappa-agreement with topics',annt_responses,tweets_topics,err[0,0])
    ##print(annt_tpc)
    (majority_voting,annt_mv_tpc)=majorityVoting(annt_responses,nb_labels,np_topics)
    err[0,2]=accuracy(majority_voting,groundTruth_temp,'Majority Voting',annt_responses,tweets_topics,err[0,2])
    err_rel[0,2]=accuracyReliability(annt_topics,annt_mv_tpc)
    ##print(annt_mv_tpc)
   
    (mv_withTopics,mv_annt_tpc)=mvWithTopics(annt_responses,np_topics,nb_labels,groundTruth_temp) 
    err[0,1]=accuracy(mv_withTopics,groundTruth_temp,'Majority Voting with Topics',annt_responses,tweets_topics,err[0,1])
    err_rel[0,1]=accuracyReliability(annt_topics,mv_annt_tpc)
    ##print(mv_annt_tpc)
    return (err,err_rel)
#===================Test===========================================================
array_nb_topics=[25]
array_nb_annotators=[5]#,10,25,30]
array_maxFeatures=[2000]
array_midDf=[1]
array_maxiter=[700]
array_meanAccuarcy=[20]#,40]
array_SdAccuracy=[10]
array_meanLikelihood=[15]#[1,5,10,15]
array_nb_tweets=[100]#,250,500,1000,5000]
nb_rounds=60


result=[]
  
for nb_t in array_nb_topics:
 for mF in array_maxFeatures:
  for mD in array_midDf:
   for m in array_maxiter:
    for sdAcc in array_SdAccuracy:
     for meanAcc in array_meanAccuarcy:
      for nb_tweets in array_nb_tweets:
       for nb_a in array_nb_annotators:
        for meanLH in array_meanLikelihood:
     
         row_csv=[]
         row_csv.append(nb_tweets)
         row_csv.append(nb_t)
         row_csv.append(nb_a)
         row_csv.append("tweets.csv")
         row_csv.append(mF)
         
        
         row_csv.append(meanAcc)
         row_csv.append(sdAcc)
         row_csv.append(meanLH)
         
         ##dataset = pd.read_csv("finalizedfull.csv")
         dataset = pd.read_csv("Tweets.csv")
         data = dataset['text']
         ##data = dataset['tweet']
         #f=[]
         #f=data1
         vectorizer = TfidfVectorizer(max_features=mF, min_df=mD, stop_words='english')
         X = vectorizer.fit_transform(data)
         nmf = NMF(n_components=nb_t, init='nndsvd', random_state=0, max_iter = m)
         np_topics1=nb_t
         W=[]
         W1=[]
         W1 = nmf.fit_transform(X)
         W=W1.copy()
         S=W.copy()
         ss=S[:nb_tweets]
         tweets_topics=[]
         groundTruth_temp=[]
         documents = dataset[['airline_sentiment','text']]
         documents.replace({'neutral': 1, 'positive': 2, 'negative': 3}, inplace=True)
         groundTruth=documents['airline_sentiment']
         twe_tpc=[]
         for i in range(0,len(ss)):
             contain=False
             for j in range(0,len(ss[0])):
                 if ss[i,j]!=0.0:
                     contain=True
             if contain:
                 twe_tpc.append(ss[i])
                 groundTruth_temp.append(groundTruth[i])
         
         tweets_topics=np.asarray(twe_tpc)
         np_tweets=len(tweets_topics)
         np_topics=len(tweets_topics[0])
         nb_annotators=nb_a
         nb_labels=3
         annt_responses=np.full((nb_a,np_tweets),0)
         annt_topics=np.full((nb_a,np_topics),1.0)
         trueLabels=[]
         topics=np.zeros(np_topics)
         cc=np.zeros((1,3))
         error_rel=np.zeros((1,3))
         avg_annotated_tweets=0
         for i in range (0,nb_rounds): 
         
             annt_topics=[]
             annt_responses=[]
             annt_topicsFloat=[]
             
             (annt_topics,annt_topicsFloat,annt_responses)=simulateLabels(groundTruth_temp,tweets_topics,topics,nb_a,meanAcc,sdAcc,meanLH,nb_labels)
             ##print(annt_topics)
            
             ##print(groundTruth_temp)
             mv=[]
             
             mv_nb=np.zeros(np_tweets)
             (mv,tp)=majorityVoting(annt_responses,nb_labels,np_topics)
             res_ord_list=[]
             annt_res_ordered=np.zeros(annt_responses.shape)
             label_ord_list=[]
             groundTruth_order=np.zeros((len(groundTruth_temp),1))
             for i in range(0,len(annt_responses[0])):
                 for j in range (0,len(annt_responses)):
                    if annt_responses[j][i]==mv[i] and mv[i]!=0:
                        mv_nb[i]=mv_nb[i]+1
             import math
             maximum=math.floor(mv_nb.max())
             ##print('mv_nb',mv_nb)
             for i in range(0,maximum):
                 for j in range(0,len(mv_nb)):
                  if mv_nb[j]==(maximum-i) and mv_nb[j]!=0:
                      res_ord_list.append(annt_responses[:,j])
                      label_ord_list.append(groundTruth_temp[j])
                      ##print(j)
             annt_res_ordered=np.asarray(res_ord_list)
             annt_res_ordered_tran=annt_res_ordered.transpose()
             groundTruth_order=np.asarray(label_ord_list)
             groundTruth_order_tran= groundTruth_order.transpose()
             if annt_res_ordered_tran.size!=0:
               avg_annotated_tweets=avg_annotated_tweets+len(annt_res_ordered_tran[0]) 
               (cc,error_rel)=mainRunWithSparsity(annt_res_ordered_tran,annt_topics,annt_topicsFloat,nb_labels,tweets_topics,groundTruth_order_tran,cc,error_rel)
               
         for counter in range(0,len(cc[0])):
             row_csv.append(math.sqrt(cc[0,counter]/nb_rounds))
             row_csv.append(math.sqrt(error_rel[0,counter]/nb_rounds))
         row_csv.append(avg_annotated_tweets/nb_rounds)
         my_file = Path('result1_file.csv')
         if my_file.exists():
             with open('result1_file.csv', 'a') as incsv:
              writer= csv.writer(incsv,lineterminator='\n')
              writer.writerow(row_csv)
              incsv.close()
         else:
             with open('result1_file.csv', 'w') as incsv:
              writer = csv.DictWriter(incsv, fieldnames = ["nb_tweets","nb_topics", "nb_annotators","Dataset","maxFeatures","Mean Accuracy","SD Accuracy","Mean Likelihood","Kappa_agreement","Accuracy of Reliability","Majorty Voting topics","Accuracy of Reliability","MV","Accuracy of Reliability","Annotated tweets"])
              writer.writeheader()
              incsv.close() 
             with open('result1_file.csv', 'a') as incsv:
              writer= csv.writer(incsv,lineterminator='\n')
              writer.writerow(row_csv)
              incsv.close()
                
         count=0
for i in range(len(annt_responses)):
    for j in range(len(annt_responses[i])):
         if annt_responses[i][j]==0:
             count=count+1
            
print ('sparsity of the responses matrix',count/(np_tweets*nb_annotators),'empty:',count,'of ',np_tweets*nb_annotators)

                    
#=============EndTest============================================================
