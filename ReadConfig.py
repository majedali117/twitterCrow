
# coding: utf-8

# In[7]:


import json
import pandas as pd


# In[8]:


with open('config.json', 'r') as f:
    config = json.load(f)

consumer_key = config['TwitterConfig']['consumer_key']
consumer_secret = config['TwitterConfig']['consumer_secret']
access_token_key = config['TwitterConfig']['access_token_key']
access_token_secret = config['TwitterConfig']['access_token_secret']

Param_No_of_Topics = config['Parameters']['No_of_Topics']
Param_No_of_Anotators = config['Parameters']['No_of_Anotators']
Param_UseSimulator = config['Parameters']['UseSimulator']

DS_FileName = config['DataSetDetails']['FileName']
DS_TweetColumnName = config['DataSetDetails']['TweetColumnName']
DS_SentimentColumn = config['DataSetDetails']['SentimentColumn']
DS_PositiveSentiment = config['DataSetDetails']['PositiveSentiment']
DS_NegativeSentiment = config['DataSetDetails']['NegativeSentiment']
DS_NeutralSentiment = config['DataSetDetails']['NeutralSentiment']

print(consumer_key)
print(Param_No_of_Topics)
print(DS_FileName)

dataset = pd.read_csv(DS_FileName)
documents = dataset[[DS_TweetColumnName,DS_SentimentColumn]]

print(documents)

