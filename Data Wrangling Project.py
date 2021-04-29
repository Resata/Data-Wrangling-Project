#!/usr/bin/env python
# coding: utf-8

# ## Gathering
# In this phase we gather the necessary data for our analysis which can be acheived using the given data sources as follows:
# 
# 1- Enhanced Twitter Archive (.csv)
# 
# 2- Twitter API.
# 
# 3- Image Predictions File. 

# In[1]:


#Importing important libraries

import numpy as np 
import pandas as pd 
import requests
import re 
import json
import tweepy
import matplotlib.pyplot as plt 
import os
import datetime 
from scipy import stats
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')
from timeit import default_timer as timer


# ### 1- Twitter Archive

# In[2]:


#Creating dataframe from twitter-archive-enhanced
archive_df = pd.read_csv('twitter-archive-enhanced.csv')
archive_df.head()


# ### 2- Image Predictions

# In[3]:


#Downloading (image_predictions.tsv) file programmatically using the requests library using the provided url
url="https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
response = requests.get(url)


# In[4]:


#Creating a folder if doesn't exist & write the file_name inside it
with open('image_predictions.tsv', 'wb') as file:
    file.write(response.content)

#Put the tsv file to a Pandas DataFrame
image_predictions = pd.read_csv('image_predictions.tsv', sep= '\t')


# In[5]:



#Now we need to find the Tweets IDs to to get our JSON data
tweet_ids = archive_df.tweet_id.values
len(tweet_ids)


# ### 3- Twitter API 

# In[6]:


#We create a for loop to add each tweet to a list
tweets_data = []
#Opening the tweets file
tweet_file = open('tweet_json.txt', "r")

for line in tweet_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
            continue 
            
tweet_file.close()
#Create a Pandas DataFrame for the formed Tweet file
tweet_info = pd.DataFrame()


# In[7]:


#Insert the required variables to the newly formed DataFrame

tweet_info["id"] = list(map(lambda tweet: tweet["id"], tweets_data))
tweet_info["favorite_count"] = list(map(lambda tweet: tweet["favorite_count"], tweets_data))
tweet_info["retweet_count"] = list(map(lambda tweet: tweet["retweet_count"], tweets_data ))
tweet_info.head()


# ## Assessing
# In this stage we display the data we gathered in the 'Gathering' step to assess its Quality and Tidiness.
# Quality dimensions or aspects are mainly:
# 
# 1- Completness (checking if there are any missing records).
# 
# 2- Validity (Checking if the values displayed are 'valid' i.e data that follow certain known rules)
# 
# 3- Accuracy (a significant decrease or increase in a value is considered an 'inaccurate data')
# 
# 4- Consistency (There should be only one way to represent or refer to a value otherwise the data is called to be 'inconsistent')

# In[8]:


#Now we review the first records in the twitter archive
archive_df.head(10)


# In[9]:


#Review the last 5 records in the same DataFrame
archive_df.tail(5)


# In[10]:


#Get the main Info of the Twitter Archive DataFrame
archive_df.info()


# In[11]:


#Get more details about the Twitter Archive DF.
archive_df.describe()


# ### 2- Image Predictions 

# In[12]:


image_predictions.head()


# In[13]:


image_predictions.tail()


# In[14]:


image_predictions.info()


# In[15]:


image_predictions.describe()


# ### 3- Twitter Info

# In[16]:


tweet_info.head()


# In[17]:


tweet_info.tail()


# In[18]:


tweet_info.info()


# In[19]:


tweet_info.describe()


# ### To make the cleaning easier we can join the tweet_info dataset to the Twitter Archive dataset.

# In[20]:


archive_df = archive_df.rename(columns={'tweet_id': 'id'})


# In[21]:


master_df = pd.merge(archive_df, tweet_info, on=['id'], how='left')
master_df.head(2)


# ## Cleaning

# ### Quality:
# 
# #### Master DF & Image_predictions
# 
# 1- There're 59 missing expanded_urls which refer to tweets without photos so we have to remove those records.
# 
# 2- There're 181 retweet records that should be removed.
# 
# 3- retweet_counts & favorite_counts columns should be in Integer form not float.
# 
# 4- timestamp should be in datetime format.
# 
# 5- Correct wrong numerator values & change 'rating_numerator' & 'rating_denominator' to float.
# 
# 6- Make the 'source' more readable and then convert it to category.
# 
# 7- Tweet_id in image_predictions and Twitter Archive should be 'str'
# 
# 8- Change missing values in the 'name' column to NaN and remove incorrect names and remove incorrect names.
# 
# 9- Correct Ratings that were extracted incorrectly from the text.
# 
# ### Tideness:
# 
# 1- Create a column 'dog_stage' instead of the 4 columns.
# 
# 2- Join Twitter_archive and tweet_info with the Image_predictions DataSet to one dataframe (Final DF).
# 

# In[22]:


#Create copies of the datasets before cleaning.
master_df_clean = master_df.copy()
image_predictions_clean = image_predictions.copy()


# ### Define
# Remove the 59 records that have no images

# ### Code

# In[23]:


master_df_clean = master_df_clean.dropna(subset=['expanded_urls'])


# ### Test

# In[24]:


master_df_clean.expanded_urls.isnull().sum()


# ### Define
# Remove the 181 retweet records

# ### Code

# In[25]:


master_df_clean = master_df_clean[master_df_clean['retweeted_status_id'].isnull()]


# 
# ### Test

# In[26]:


master_df_clean.info()


# ### Code

# In[27]:


columns = ['retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp']
master_df_clean = master_df_clean.drop(columns, axis=1)


# ### Test

# In[28]:


master_df_clean.info()


# ### Define
# convert the 'NaN values in the  favorite count & retweet columns
# 
# change their data type from float to integer

# ### Code
# 

# In[29]:


master_df_clean['favorite_count'].fillna(0, inplace=True)
master_df_clean['retweet_count'].fillna(0, inplace=True)


# In[30]:


master_df_clean.favorite_count = master_df_clean.favorite_count.astype(int)


# ### Test
# 

# In[31]:


master_df_clean.favorite_count.isnull().sum()


# In[32]:


master_df_clean.retweet_count.isnull().sum()


# In[33]:


master_df_clean.info()


# ### Define
# Change the timestamp format from string to datetime

# ### Code

# In[34]:


master_df_clean['timestamp'] = master_df_clean['timestamp'].apply(pd.to_datetime)


# ### Test

# In[35]:


master_df_clean.info()


# ### Define
# Change 'rating_numerator' & 'rating_denominator' to float.
# 
# Modify tweets having incorrect numerators.
# 
# Edit the tweet that has no rating.

# In[36]:


#Find tweets with decimal ratings.
master_df_clean[master_df_clean.text.str.contains(r"(\d+\.\d*\/\d+)")]
#It seems that rating numerator for these records doesn't make sense which means that the ratings weren't extracted correctly.


# ### Code

# In[37]:


#Edit incorrect numerators
master_df_clean.loc[(master_df_clean['id'] == 680494726643068929) & (master_df_clean['rating_numerator'] == 26), ['rating_numerator']] = 11.26
master_df_clean.loc[(master_df_clean['id'] == 786709082849828864) & (master_df_clean['rating_numerator'] == 75), ['rating_numerator']] = 9.75
master_df_clean.loc[(master_df_clean['id'] == 883482846933004288) & (master_df_clean['rating_numerator'] == 5), ['rating_numerator']] = 13.5
master_df_clean.loc[(master_df_clean['id'] == 778027034220126208) & (master_df_clean['rating_numerator'] == 27), ['rating_numerator']] = 11.27


# ### Test

# In[38]:


master_df_clean.loc[(master_df_clean['id'] == 680494726643068929)]


# In[39]:


master_df_clean.loc[(master_df_clean['id'] == 786709082849828864)]


# In[40]:


master_df_clean.loc[(master_df_clean['id'] == 778027034220126208)]


# In[41]:


master_df_clean.loc[(master_df_clean['id'] == 883482846933004288)]


# ### Code

# In[42]:


#Change the rating format into float.
master_df_clean.rating_denominator = master_df_clean.rating_denominator.astype(float)
master_df_clean.rating_numerator = master_df_clean.rating_numerator.astype(float)


# ### Test

# In[43]:


master_df_clean.info()


# ### Code

# In[44]:


#Find the tweet that has incorrect extracted rating
master_df_clean.loc[(master_df_clean['id'] == 810984652412424192)]


# In[45]:


#Removing the incorrect rating
master_df_clean.rating_numerator[516] = ''
master_df_clean.rating_denominator[516] = ''


# ### Test

# In[46]:


master_df_clean.loc[(master_df_clean['id'] == 810984652412424192)]


# ### Code

# In[47]:


#I noticed  1/2 rating which is impossible, no one would give this rating, the rating must've been extracted wrong.
master_df_clean.loc[(master_df_clean['id'] == 666287406224695296)]
master_df_clean.text[2335]
#It's clear that the correct rating is '9/10' not '1/2'


# In[48]:


#Correct the wrong rating for (666287406224695296)
master_df_clean.rating_numerator[2335] = '9'
master_df_clean.rating_denominator[2335] = '10'


# ### Test

# In[49]:


master_df_clean.loc[(master_df_clean['id'] == 666287406224695296)]


# ### Code

# In[50]:


# I noticed a 1/10 rating which is very low, let's check it.
master_df_clean.loc[(master_df_clean['id'] == 670783437142401025)]
master_df_clean.text[2091]
# That's weird ! who gives a dog 1/10?! I thought the rating wasn't extracted correctly but that's not the case. 
# it's not a dog, it's a caterpillar!


# In[51]:


# I also noticed a 2/10 rating which is very low:
master_df_clean.loc[(master_df_clean['id'] == 670826280409919488)]
master_df_clean.text[2079]
# There's nothing wrong with the rating extraction, this rating is also very low which makes sense because it's not a dog!


# ### Define
# Make the 'source' more readable and convert its format into 'category'

# ### Code

# In[52]:


master_df_clean['source'] = master_df_clean['source'].str.extract('^<a.+>(.+)</a>$')


# ### Test

# In[53]:


master_df_clean.source.value_counts()


# ### Code

# In[54]:


master_df_clean.source = master_df_clean.source.astype('category')


# ### Test

# In[55]:


master_df_clean.info()


# ### Define
# Change tweet_id to 'str' in Image Predictions DataFrame

# ### Code

# In[56]:


image_predictions_clean.tweet_id = image_predictions_clean.tweet_id.astype('str')


# ### Test
# 

# In[57]:


image_predictions_clean.info()


# ### Define
# remove incorrect dogs names

# ### Code

# In[58]:


lower_names= master_df_clean[master_df_clean.name.str.islower()]['id']
for x in lower_names:
    for id_to_remove in master_df_clean.index[master_df_clean['id'] == x].tolist():
        master_df_clean = master_df_clean.drop(id_to_remove)


# ## Test

# In[59]:


master_df_clean[master_df_clean.name.str.islower()]


# ## Define
# Create a 'dog stage' column instead of the 4 classification columns.
# 
# Merge the multiple dog stages.

# ## Code

# In[60]:


### Find dogs with multiple stages
master_df_clean.groupby(["doggo", "floofer", "pupper", "puppo"]).size().reset_index().rename(columns={0: "count"})


# In[61]:


# Remove the 'NaN' or 'None' values
master_df_clean.doggo.replace('None', '', inplace=True)
master_df_clean.doggo.replace(np.NaN, '', inplace=True)

master_df_clean.floofer.replace('None', '', inplace=True)
master_df_clean.floofer.replace(np.NaN, '', inplace=True)

master_df_clean.pupper.replace('None', '', inplace=True)
master_df_clean.pupper.replace(np.NaN, '', inplace=True)

master_df_clean.puppo.replace('None', '', inplace=True)
master_df_clean.puppo.replace(np.NaN, '', inplace=True)


# In[62]:


#Extract the dog's stage from the text
master_df_clean['dog_stage'] = master_df_clean['text'].str.extract('(puppo|pupper|floofer|doggo)', expand=True)


# In[63]:


#Create multiple dog stage
master_df_clean['dog_stage'] = master_df_clean.doggo + master_df_clean.floofer + master_df_clean.pupper + master_df_clean.puppo
master_df_clean.loc[master_df_clean.dog_stage == 'doggopupper', 'dog_stage'] = 'doggo, pupper'
master_df_clean.loc[master_df_clean.dog_stage == 'doggopuppo', 'dog_stage'] = 'doggo, puppo'
master_df_clean.loc[master_df_clean.dog_stage == 'doggofloofer', 'dog_stage'] = 'doggo, floofer'


# In[64]:


#Drop the 4 classification columns.
columns = ['doggo', 'floofer', 'pupper', 'puppo']
master_df_clean = master_df_clean.drop(columns, axis=1)


# ## Test

# In[65]:


master_df_clean.sample(50)


# ### Define
# Merge the Image Predictions Dataset with the Master DataFrame.

# In[66]:


image_predictions_clean = image_predictions_clean.rename(columns={'tweet_id': 'id'})


# In[67]:


master_df_clean.info()


# In[68]:


master_df_clean.id = master_df_clean.id.astype('str')


# ### Code

# In[69]:


final_df = master_df_clean.merge(image_predictions_clean, on= 'id', how= 'inner')


# ### Test

# In[81]:


final_df.sample(50)


# ### Storing

# In[71]:


final_df.to_csv('master_df.csv')


# ### Analysis & Visualization

# ### 1- Dog Stages

# In[82]:


final_df.dog_stage.value_counts()


# In[73]:


dog_stage = ['pupper', 'doggo', 'puppo', 'doggo & pupper ', 'floofer', 'doggo & puppo', 'doggo & floofer']
no_dogs = [194, 61, 22, 8, 7, 1, 1]

fig,ax = plt.subplots(figsize = (12,6))
plt.bar(dog_stage, no_dogs, width= 0.6)
plt.title("Number of dogs in each stage")
plt.xlabel('Dog Stage')
plt.ylabel('Count')


# ### 2- Common Sources

# In[83]:


final_df.source.value_counts()


# In[85]:


source_name= ['Twitter for iPhone', 'Twitter Web Client ', 'TweetDeck', 'Vine', ]
number= [1861, 25, 10, 0]
fig,ax = plt.subplots(figsize = (12,6))
plt.bar(source_name, number, width= 0.5)
plt.title("Most Common Sources Of Tweets")
plt.xlabel("Source")
plt.ylabel("Number")


# In[76]:


final_df.retweet_count.describe()


# In[77]:


final_df.favorite_count.describe()


# In[78]:


final_df.describe()


# ### Insights
# The most common Dog Stage is 'Pupper' and the least is 'Doggo & Floofer' and 'Doggo & Puppo.
#  
# The most common source of tweets is 'Twitter for iPhone' with a number of 1862 while the least is ' TweetDeck'  with 10 times.
# 
# The Mean Retweet Count is 2801.
#  
# The Mean Favorite Count is 9043.
#  
# The Minimum Image Number is (1) & The Maximum is (4).
# 
# There're some incorrect ratings I noticed, some of them were extracted incorrectly from the text, others were unfortunately extracted right but recieved low values such as '1/10', '2/10', '5/10'.
# 

# In[ ]:




