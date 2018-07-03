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
import json

#=============================Export annotators================================================

def exportMarix(fileName,matrix):
 with open(fileName, "w") as file:
        writer = csv.writer(file,lineterminator='\n')
        writer.writerows(matrix) 
def loadMatrix(fileName):
 matrix=[]
 matrix = np.genfromtxt (fileName, delimiter=",")
 return matrix

def simulateLabels(groundTruth,tweets_topics,topics,numberAnnotators,meanAccuarcy,SdAccuracy,meanLikelihood,nb_labels):
  x=[]
  likelihood=[]
  numberTopics=len(tweets_topics[0])
  numberTweets=len(tweets_topics)
  annt_responses=np.full((numberAnnotators,np_tweets),0)
  annt_topics=np.full((numberAnnotators,np_topics),1.0)
  for m in range(0,numberAnnotators):
   done=False
   while(done==False):
    ##val=np.random.normal(meanAccuarcy, SdAccuracy,1)#the accurcy for each annotater
    ##perfect val=skewnorm.rvs(0.1,loc=0.3, size=1)
    val=skewnorm.rvs(SdAccuracy,loc=meanAccuarcy, size=1)
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
             
      annt_counter.append(counter)
      for l in range (0,numberTopics):
          annt_topics[m,l]=annt_topics[m,l]/annt_counter[m]
          
  return (annt_topics,annt_responses)

'''
#===========generate labels===============================       
def fillMissingValues(annt_responses):
        annotator_responses=[]
        annotator_responses=annt_responses.copy()
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
                text_clf = text_clf.fit( np.asarray(trainingSet),trainingTarget)
                for j in range(0,np_tweets):
                     if annotator_responses[i][j]== 0:
                       test=[]
                       test.append(tweets_topics[j])
                       predicted = text_clf.predict(test)
                       annotator_responses[i][j]=predicted[0]
                       if annotator_responses[i][j]==groundTruth[j]:
                            accuracy=accuracy+1
                       f=f+1
        ##print('annotator_responses')
        return annotator_responses
#==========================inter-agreement===========================================
'''
def inter_agreement(annotators_responses,numberTopics):
  nb_annotators=len(annotators_responses)
  np_tweets=len(annotators_responses[0])
  agree=[]
  
  for i in range(0,nb_annotators):
    agreement=0.0
    common_annt=0
    
    for j in range(0,nb_annotators):
      common=False
      for z in range(0,np_tweets):
        if annt_responses[i][z]!=0:
         if annt_responses[i][z]==annt_responses[j][z] and i!=j:
           agreement=agreement+1
           common=True
      if common:
            common_annt=common_annt+1
            common=False
    if common_annt==0:
        common_annt=1
    ##print(i,agreement,common_annt)
    agree.append(agreement/(np_tweets*common_annt))
  ##print('agree',agree)
  return agree
#===============End inter-agreement==================================
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
        for i in range(0,nb_labels):
          if total!=0.0:
            pre=pre+(cols[i]*rows[i])/total
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
#=====================inter agreement with topics
#
def interAgreementWithTopics(annt_responses,np_topics,nb_labels,groundTruth):
    agree=inter_agreement(annt_responses,np_topics)
    trueLabels=[]
    annt_tpc_counter=np.zeros((len(annt_responses),np_topics))
    np_tweets=len(annt_responses[0])
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
                   ##sim=sim+(agree[j]*annt_tpc[j,c])
                   accm=accm+(agree[j]*annt_tpc[j,c])
                 ##counter=counter+1
         if counter!=0:
          sim=accm/counter
         if highsim<sim:
             truelabel=label
             highsim=sim
     '''if(truelabel!=groundTruth[i]):
      print('missed',i,truelabel,groundTruth[i],highsim,counter)
      print('res',annt_responses[:,i])
      for d in range(0,np_topics):
         if tweets_topics[i,d]!=0.0:
             print(annt_tpc[:,d])
        '''     
     trueLabels.append(truelabel)
     for j in range(0,nb_annotators):
       if (annt_responses[j][i]!=0): 
          if (annt_responses[j][i]==trueLabels[i]) :
                  for c in range(0,np_topics):
                      if tweets_topics[i,c]!=0.0:
                        annt_tpc[j,c]=round(annt_tpc[j,c]+(tweets_topics[i,c]),4)
          '''else:
                 for c in range(0,np_topics):
                      if tweets_topics[i,c]!=0.0 and annt_responses[j][i]!=0 :
                        annt_tpc[j,c]=round(annt_tpc[j,c]-(tweets_topics[i,c]),4)'''
    ##print(trueLabels)
    ##print(groundTruth)
    ##print(annt_responses)
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
      if (annt_responses[j][i]==truelabel) :
              for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0:
                    annt_tpc_kappa[j,c]=round(annt_tpc_kappa[j,c]+(tweets_topics[i,c]),4)
      else:
             for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0 and annt_responses[j][i]!=0 :
                    annt_tpc_kappa[j,c]=round(annt_tpc_kappa[j,c]-(tweets_topics[i,c]),4)
    ##print('kappTopics',Kappa_trueLabels)
    ##print(annt_tpc_kappa)
    return(Kappa_trueLabels,annt_tpc_kappa)
#==============End Kappa with topics=================================
#=======================inter agreement without topics===============================
def interAgreementWithoutTopics(annt_responses,nb_labels):
    trueLabelsWithoutTopics=[]
    np_topics=len(tweets_topics[0])
    np_tweets=len(annt_responses)
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
    ##print('interWithoutTopics',trueLabelsWithoutTopics)
    return trueLabelsWithoutTopics
#======================End inter agreement without topics
def reliability_interAgreement(tweets_topics,annt_responses,topics,nb_labels):
    nb_annotators=len(annt_responses)
    np_topics=len(tweets_topics[0])
    annt_tpc2=np.full((nb_annotators,np_topics),1.0)
    trueLabels=[]
    trueLabels=interAgreementWithoutTopics(annt_responses,nb_labels)
    agree=inter_agreement(annt_responses,np_topics)
    for j in range(0,len(annt_responses)):
      for i in range(0,len(annt_responses[0])):
          if (annt_responses[j][i]==trueLabels[i]) :
                  for c in range(0,np_topics):
                      if tweets_topics[i,c]!=0.0:
                        annt_tpc2[j,c]=round(annt_tpc2[j,c]+(tweets_topics[i,c]),4)
          else:
                 for c in range(0,np_topics):
                      if tweets_topics[i,c]!=0.0 and annt_responses[j][i]!=0:
                        annt_tpc2[j,c]=round(annt_tpc2[j,c]-(tweets_topics[i,c]),4)

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
    ##print('rel_inter',trueLabels_agree_rel)
    ##print(annt_tpc2)
    return(trueLabels_agree_rel)
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
    ##print('kappaWithoutTopics',Kappa_trueLabelsWithoutTopics)
    return Kappa_trueLabelsWithoutTopics
#==============End Kappa without topics=================================
#====================Kappa interagreement then reliability==================
def reliability_kappaInterAgreement(tweets_topics,annt_responses,nb_labels):
    np_topics=len(tweets_topics[0])
    Kappa_annt_tpc2=[]
    Kappa_annt_tpc2=np.full((nb_annotators,np_topics),1.0)
    kappa_agree=kappa_aggreement(annt_responses,nb_labels)
    np_tweets=len(annt_responses[0])
    Kappa_trueLabelsWithoutTopics=kappaInterAgreementWithoutTopics(annt_responses,nb_labels)
    for j in range(0,nb_annotators):
     for i in range(0,np_tweets):
      if (annt_responses[j][i]==Kappa_trueLabelsWithoutTopics[i]) :
              for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0:
                    Kappa_annt_tpc2[j,c]=round(Kappa_annt_tpc2[j,c]+(tweets_topics[i,c]),4)
      else:
             for c in range(0,np_topics):
                  if tweets_topics[i,c]!=0.0 and annt_responses[j][i]!=0:
                    Kappa_annt_tpc2[j,c]=round(Kappa_annt_tpc2[j,c]-(tweets_topics[i,c]),4)
    
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
    ##print('rel_kappa',kappa_trueLabels_agree_rel)
    ##print(Kappa_annt_tpc2)
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
    ##print('MV',majority_voting)
    return majority_voting

def reliability_majority_voting(majority_voting,tweets_topics,annt_responses,topics,nb_labels):
    nb_annotators=len(annt_responses)
    np_topics=len(tweets_topics[0])
    annt_tpc2=[]
    annt_tpc2=np.full((nb_annotators,np_topics),1.0)
    trueLabels=[]
    np_tweets=len(annt_responses[0])
    trueLabels=majority_voting.copy()
    for j in range(0,len(annt_responses)):
      for i in range(0,len(annt_responses[0])):
          if (annt_responses[j][i]==trueLabels[i]) :
                  for c in range(0,np_topics):
                      if tweets_topics[i,c]!=0.0:
                        annt_tpc2[j,c]=annt_tpc2[j,c]+1
          '''else:
                 for c in range(0,np_topics):
                      if tweets_topics[i,c]!=0.0 and annt_responses[j][i]!=0:
                        annt_tpc2[j,c]=annt_tpc2[j,c]-1
'''
    trueLabels_agree_rel=[]
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
                   ##sim=sim+annt_tpc2[j,c]
                   accm=accm+(annt_tpc2[j,c])
         if counter!=0:
          sim=accm/counter
         if highsim<sim:
             truelabel=label
             highsim=sim
     trueLabels_agree_rel.append(truelabel)
    ##print('rel_mv',trueLabels_agree_rel)
    ##print(annt_tpc2)
    return(trueLabels_agree_rel)

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
     '''if(truelabel!=groundTruth[i]):
      print('missed here',i,truelabel,groundTruth[i],highsim,counter)
      print('res here',annt_responses[:,i])
      for d in range(0,np_topics):
         if tweets_topics[i,d]!=0.0:
             print(annt_tpc[:,d])
     '''
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
    ##print('mvTopics',trueLabels)
    ##print(annt_tpc)
    return (trueLabels,annt_tpc)
 
def accuracy(trueLabels,groundTruth,text,annt_responses,annt_tpc,tweet_topic):
    hits=0
    np_tweets=len(trueLabels)
    for i in range(0,np_tweets):
     if groundTruth[i]==trueLabels[i]:
        hits=hits+1
     '''else:
         print('true',groundTruth[i],'label',trueLabels[i])
         print(annt_responses[:,i])
         print(tweet_topic[i])
         print(i)
         for j in range(len(tweet_topic[0])):
             if tweet_topic[i][j]!=0:
                 print(annt_tpc[:,j])
    '''
    print('accuracy',text,(float(hits)/float(np_tweets)),'missclasification: ',np_tweets-hits,'of',np_tweets)
    return (float(hits)/float(np_tweets))
#================================================================================
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
        counter=0
        for i in range(0,nb_annotators):
            for j in range(0,np_topics):
                if unbiased_annt_tpcs[i][j]!=unbiased_annt_topics[i][j]:
                    counter=counter+1
        print('accuarcy of annotators reliability',counter/(nb_annotators*np_topics))
#========================accuracy======================================
      
def mainRunWithSparsity(annt_responses,annt_topics,nb_labels,tweets_topics,groundTruth_temp,row_csv):
    np_topics=len(annt_topics[0])
    (trueLabels,annt_tpc)=interAgreementWithTopics(annt_responses,np_topics,nb_labels,groundTruth_temp)
    (Kappa_trueLabels,annt_tpc_kappa)=kappaInteragreemtWithTopics(annt_responses,nb_labels,tweets_topics)
    trueLabelsWithoutTopics=interAgreementWithoutTopics(annt_responses,nb_labels)
    Kappa_trueLabelsWithoutTopics=kappaInterAgreementWithoutTopics(annt_responses,nb_labels)
    #kappa_trueLabels_agree_rel= reliability_kappaInterAgreement(tweets_topics,annt_responses,nb_labels)
    majority_voting=majorityVoting(annt_responses,nb_labels)
    
    rel_MV=reliability_majority_voting(majority_voting,tweets_topics,annt_responses,topics,nb_labels)
    (mv_withTopics,mv_annt_tpc)=mvWithTopics(annt_responses,np_topics,nb_labels,groundTruth_temp) 
    row_csv.append(accuracy(trueLabels,groundTruth_temp,'inter-agreement with topics',annt_responses,annt_tpc,tweets_topics))
    compareReliability(annt_topics,annt_tpc)
    ##row_csv.append(accuracy(trueLabelsWithoutTopics,groundTruth_temp,'inter-agreement without topics',annt_responses,annt_tpc,tweets_topics))
   # row_csv.append(accuracy(trueLabels_agree_rel,groundTruth_temp,'inter-agreement then reliability'))
    ##row_csv.append(accuracy(Kappa_trueLabelsWithoutTopics,groundTruth_temp,'Kappa without Topics',annt_responses,annt_tpc_kappa,tweets_topics))
    ##row_csv.append(accuracy(Kappa_trueLabels,groundTruth_temp,'Kappa Topics',annt_responses,annt_tpc_kappa,tweets_topics))
    compareReliability(annt_topics,annt_tpc_kappa)
    #row_csv.append(accuracy(kappa_trueLabels_agree_rel,groundTruth_temp,'Kappa then reliability'))
    row_csv.append(accuracy(majority_voting,groundTruth_temp,'Majority Voting',annt_responses,mv_annt_tpc,tweets_topics))
    row_csv.append(accuracy(mv_withTopics,groundTruth_temp,'Majority Voting with Topics',annt_responses,mv_annt_tpc,tweets_topics))
    compareReliability(annt_topics,mv_annt_tpc)
    row_csv.append(accuracy(rel_MV,groundTruth_temp,'Majority Voting then reliability',annt_responses,mv_annt_tpc,tweets_topics))
    
    
#===================End Majority Voting===ُ================================
'''exportResponses('sparseMatrix.csv',annt_responses)
exportResponses('fullMatrix.csv',annotator_responses)

print('arr',arr.shape)
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

'''
#===================Test===========================================================
'''array_nb_topics=[5,10,15,20,25]
array_nb_annotators=[5,10,15,20,25,50]
array_maxFeatures=[100,500,1000,1500,2000]
array_midDf=[1,5,10,20,50,100]
array_maxiter=[100,200,500,1000]
array_meanAccuarcy=[0.5,0.7]
array_SdAccuracy=[0.2,0.1]
array_meanLikelihood=[1.0,1.5]
'''
''' good results
array_nb_topics=[20]
array_nb_annotators=[20]
array_maxFeatures=[500]
array_midDf=[1]
array_maxiter=[700]
array_meanAccuarcy=[0.1]
array_SdAccuracy=[0.1]
array_meanLikelihood=[0.1]
'''

''' good results 
array_nb_topics=[20]
array_nb_annotators=[10]
array_maxFeatures=[500]
array_midDf=[25]
array_maxiter=[700][1000]
array_meanAccuarcy=[0.1]
array_SdAccuracy=[0.1]
array_meanLikelihood=[3]

array_nb_topics=[20]
array_nb_annotators=[10]
array_maxFeatures=[500]
array_midDf=[25]
array_maxiter=[700]
array_meanAccuarcy=[0.1]
array_SdAccuracy=[0.1]
array_meanLikelihood=[1]


array_nb_topics=[20]
array_nb_annotators=[10]
array_maxFeatures=[500]
array_midDf=[45]
array_maxiter=[700]
array_meanAccuarcy=[0.1]
array_SdAccuracy=[0.1]
array_meanLikelihood=[3]


array_nb_topics=[20]
array_nb_annotators=[10]
array_maxFeatures=[500]
array_midDf=[45]
array_maxiter=[700]
array_meanAccuarcy=[0.08]
array_SdAccuracy=[0.1]
array_meanLikelihood=[0.08]

array_nb_topics=[20]
array_nb_annotators=[10]
array_maxFeatures=[500]
array_midDf=[45]
array_maxiter=[700]
array_meanAccuarcy=[0.03]
array_SdAccuracy=[0.1]
array_meanLikelihood=[0.08]

similar performance to MV
array_nb_topics=[20][10]
array_nb_annotators=[10]
array_maxFeatures=[500][100]
array_midDf=[45][25]
array_maxiter=[700]
array_meanAccuarcy=[0.03]
array_SdAccuracy=[0.1]
array_meanLikelihood=[0.02

airelines mv_rel best
array_nb_topics=[10][20]
array_nb_annotators=[10]
array_maxFeatures=[100]
array_midDf=[25][10]
array_maxiter=[700]
array_meanAccuarcy=[0.01]
array_SdAccuracy=[0.05]
array_meanLikelihood=[5][0.1]


array_nb_topics=[20]
array_nb_annotators=[20]
array_maxFeatures=[2000]
array_midDf=[25]
array_maxiter=[700]
array_meanAccuarcy=[0.01]
array_SdAccuracy=[0.02]
array_meanLikelihood=[0.05]




array_nb_topics=[20]
array_nb_annotators=[20]
array_maxFeatures=[2000]
array_midDf=[25]
array_maxiter=[700]
array_meanAccuarcy=[0.01]
array_SdAccuracy=[0.02]
array_meanLikelihood=[20]
tweets=500
not counting the missed labels from reliability




skew dist 10000 airlines
array_nb_topics=[20]
array_nb_annotators=[20]
array_maxFeatures=[2000]
array_midDf=[25]
array_maxiter=[700]
array_meanAccuarcy=[0.01]
array_SdAccuracy=[0.02]
array_meanLikelihood=[100]

skewed dist airlines 1000
array_nb_topics=[20]
array_nb_annotators=[20]
array_maxFeatures=[2000]
array_midDf=[25]
array_maxiter=[700]
array_meanAccuarcy=[0.01]
array_SdAccuracy=[0.02]
array_meanLikelihood=[100]




skew dist tweet 998
array_nb_topics=[20]
array_nb_annotators=[20]
array_maxFeatures=[2000]
array_midDf=[1]
array_maxiter=[700]
array_meanAccuarcy=[0.01]
array_SdAccuracy=[0.02]
array_meanLikelihood=[100]

skew dist tweet 998 significnt improve
array_nb_topics=[25]
array_nb_annotators=[20]
array_maxFeatures=[2000]
array_midDf=[1]
array_maxiter=[700]
array_meanAccuarcy=[0.01]
array_SdAccuracy=[0.02]
array_meanLikelihood=[100]

array_nb_topics=[4]
array_nb_annotators=[4]
array_maxFeatures=[2000]
array_midDf=[1]
array_maxiter=[700]
array_meanAccuarcy=[0.01]
array_SdAccuracy=[0.02]
array_meanLikelihood=[100]

'''
 
with open('config.json', 'r') as f:
    config = json.load(f)

array_nb_topics=config['Parameters']['No_of_Topics']
array_nb_annotators= config['Parameters']['No_of_Anotators']
array_nb_tweets= config['Parameters']['No_of_Tweets']

array_maxFeatures= config['MFParameters']['MaxFeatures']
array_midDf=config['MFParameters']['MidDf']
array_maxiter=config['MFParameters']['MaxIter']
array_meanAccuarcy=config['MFParameters']['MeanAccuarcy']
array_SdAccuracy=config['MFParameters']['SdAccuracy']
array_meanLikelihood=config['MFParameters']['MeanLikelihood']

DS_FileName = config['DataSetDetails']['FileName']
DS_TweetColumnName = config['DataSetDetails']['TweetColumnName']
DS_SentimentColumn = config['DataSetDetails']['SentimentColumn']
DS_PositiveSentiment = config['DataSetDetails']['PositiveSentiment']
DS_NegativeSentiment = config['DataSetDetails']['NegativeSentiment']
DS_NeutralSentiment = config['DataSetDetails']['NeutralSentiment']

DS_Result_FileName = config['DataSetDetails']['Result_FileName']

result=[]
         
for nb_t in array_nb_topics:
 for nb_a in array_nb_annotators:
  for mF in array_maxFeatures:
   for mD in array_midDf:
    for m in array_maxiter:
     for meanAcc in array_meanAccuarcy:
      for sdAcc in array_SdAccuracy:
       for meanLH in array_meanLikelihood:
         row_csv=[]
         row_csv.append(nb_t)
         row_csv.append(nb_a)
         row_csv.append(mF)
         row_csv.append(mD)
         row_csv.append(m)
         row_csv.append(meanAcc)
         row_csv.append(sdAcc)
         row_csv.append(meanLH)
         row_csv0 = []
         row_csv0.append("No of Topics")
         row_csv0.append("No of Anotators")
         row_csv0.append("Max Features")
         row_csv0.append("MidDf")
         row_csv0.append("MaxIter")
         row_csv0.append("MeanAccuarcy")
         row_csv0.append("SdAccuracy")
         row_csv0.append("MeanLikelihood")
         row_csv0.append("Inter-Aggrement with topics Accuracy")
         row_csv0.append("Majority Voting Accuracy")
         row_csv0.append("Majority Voting with topics Accuracy")
         row_csv0.append("Majority Voting then reliability Accuracy")

         dataset = pd.read_csv(DS_FileName)
         data = dataset[DS_TweetColumnName]
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
         #tweets_topics=W.copy()
         S=W.copy()
         ss=S[:array_nb_tweets]
         tweets_topics=[]
         groundTruth_temp=[]
         
         sum_rows=np.full(len(W),0.0)
         sum_rows=W.sum(axis=1)
         '''tweets_topics=np.full((6,np_topics1),0.0)   
         for i in range(6):
          for j in range(len(W[0])):
           if (sum_rows[i]!=0.0):
            tweets_topics[i][j]=(W[i,j]/sum_rows[i])
           else:
            tweets_topics[i][j]=0
         '''
         documents = dataset[[DS_TweetColumnName,DS_SentimentColumn]]
         documents.replace({DS_NeutralSentiment: 1, DS_NegativeSentiment: 2, DS_PositiveSentiment: 3}, inplace=True)
         groundTruth=documents[DS_SentimentColumn]
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
         '''for i in range(0,np_topics):
            for j in range(0,np_tweets):
                topics[i]=topics[i]+tweets_topics[j,i]
         normalizingFactor=topics.sum()
         ##print(tweets_topics[:6])
         '''
         annt_topics=[]
         annt_responses=[]
         (annt_topics,annt_responses)=simulateLabels(groundTruth_temp,tweets_topics,topics,nb_a,meanAcc,sdAcc,meanLH,nb_labels)
         ##print(annt_topics)
         ##print(annt_responses)
         ##print(groundTruth_temp)
         mv=[]
         mv_nb=np.zeros(np_tweets)
         mv=majorityVoting(annt_responses,nb_labels)
         res_ord_list=[]
         annt_res_ordered=np.zeros(annt_responses.shape)
         label_ord_list=[]
         groundTruth_order=np.zeros((len(groundTruth_temp),1))
         for i in range(0,len(annt_responses[0])):
             for j in range (0,len(annt_responses)):
                if annt_responses[j,i]==mv[i] and mv[i]!=0:
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
         ##print(annt_res_ordered_tran)
         ##print(groundTruth_order_tran)
         ##mainRunWithSparsity(annt_responses,annt_topics,nb_labels,tweets_topics,groundTruth_temp,row_csv)
         mainRunWithSparsity(annt_res_ordered_tran,annt_topics,nb_labels,tweets_topics,groundTruth_order_tran,row_csv)

         result.append(row_csv0)
         result.append(row_csv)
         
         count=0
for i in range(len(annt_responses)):
    for j in range(len(annt_responses[i])):
         if annt_responses[i][j]==0:
             count=count+1
            
print ('sparsity of the responses matrix',count/(np_tweets*nb_annotators),'empty:',count,'of ',np_tweets*nb_annotators)


with open(DS_Result_FileName, "w") as file:
  writer = csv.writer(file,lineterminator='\n')
  writer.writerows(result)     
                    
#=============EndTest============================================================
