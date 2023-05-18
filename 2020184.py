#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


df = pd.read_csv("tweets.csv")
data = df.head(100)
data


# In[94]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import re
import string
from collections import Counter
import plotly.express as px
from nltk.corpus import stopwords
import spacy


# In[97]:


temp = data.groupby('hashtags').count()['text'].reset_index().sort_values(by='text', ascending=False)  
temp


# In[109]:


plt.figure(figsize=(6, 3))
sns.countplot(x='hashtags', data=data)


# In[110]:


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    
    return float(len(c))/len(a) + len(b) - len(c)


# In[111]:


results_jaccard = []

for ind, row in data.iterrows():
    sent1 = row.text
    sent2 = row.selected_text
    
    jaccard_score = jaccard(sent1, sent2)
    results_jaccard.append([sent1, sent2, jaccard_score])


# In[ ]:





# In[ ]:





# In[7]:


#pip install emoji


# In[8]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from emoji import demojize
import emoji


# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# In[14]:


# Define a list to store cleaned tweets
cleaned_tweets = []
final_tokens = []
stemmed =[] #tokens
lemmatized = []#tokens
lemma_cleaned_tweets =[]
stem_cleaned_tweets = []

# Convert the 'text' column to a string variable
tweets = data['text'].to_string(index=False).split('\n')
  
  #Calc length of total tweets 
length=len(tweets)


for i in range(length):
    # Loop through each tweet and apply the preprocessing steps
    tweet=tweets[i]

    # Remove duplicate rows 
    tweet = re.sub(r'^RT[\s]+@[^\s]+[\s]+', '', tweet)

    # Tokenization
    tokens = nltk.word_tokenize(tweet)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if not token.lower() in stop_words]

    # Stemming 
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    for j in stemmed_tokens:
        stemmed.append(j)
    stem_tweet = ' '.join(stemmed_tokens)
    stem_cleaned_tweets.append(stem_tweet)
      
    #lemmatization
    wnl = WordNetLemmatizer()
    lemmatized_tokens = [wnl.lemmatize(token) for token in filtered_tokens]
    for j in lemmatized_tokens:
        lemmatized.append(j)
    lemma_tweet = ' '.join(lemmatized_tokens)
    lemma_cleaned_tweets.append(lemma_tweet)

    # Remove punctuations
    tweet = re.sub(r'[^\w\s]','', tweet)

    # Remove URLs
    tweet = re.sub(r'https?:\/\/[^\s]*', '', tweet)

    # Convert emojis to words
    #tweet = demojize(tweet)
    #tweet = emoji.demojize(tweet, delimiters=(' ', ' '))
    #emoji_pattern = r'/[x{1F601}-x{1F64F}]/u'
    #tweet = re.sub(emoji_pattern, '', tweet)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet) # no emoji

    #Remove Special Characters 
    tweet = re.sub(r'[^A-Za-z0-9 ]+', '', tweet)
    
    # Remove collection words
    collection_words = ['username', 'sample', 'tweet']
    fin_tokens = [token for token in lemmatized_tokens if not token.lower() in collection_words]
      
    # Join tokens back into a string
    final_tweet = ' '.join(fin_tokens)

    # Add cleaned tweet to the list
    cleaned_tweets.append(final_tweet)

      # Join tokens back into a string
      #final_tweet = ' '.join(final_tokens)
      #final_token.append(final_tokens)
    for j in fin_tokens:
        final_tokens.append(j)
final_tokens


# In[73]:


#task2(length)


# In[ ]:


#Task 3


# In[15]:


#ngrams 
def ngrams(final_tokens, n):
    n_grams=[]
    for i in range(len(final_tokens)-n+1):
        n_grams.append(' '.join(final_tokens[i:i+n]))
    
    return n_grams


# In[16]:


#Bigrams
ngrams(final_tokens,2)


# In[17]:


#Trigrams
ngrams(final_tokens,3)


# In[18]:


from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import requests
pic = np.array(Image.open(requests.get('http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png',stream=True).raw))
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, mask = pic, 
                min_font_size = 10).generate(final_tokens)


# In[55]:


#Task4 


# In[56]:


#pip install textblob


# In[19]:


import matplotlib.pyplot as plt
from textblob import TextBlob

#TEXTBLOB ON STEMMING+LEMMA
tweets = cleaned_tweets
# Perform sentiment analysis on each tweet
polarity_scores = []
for text in tweets:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    polarity_scores.append(polarity)


# Plot the sentiment scores as a bar chart
fig, ax = plt.subplots()
ax.bar(['Negative', 'Neutral', 'Positive'], [
      len(list(filter(lambda x: x < 0, polarity_scores))),
      len(list(filter(lambda x: x == 0, polarity_scores))),
      len(list(filter(lambda x: x > 0, polarity_scores)))
      ])
ax.set_title('Sentiment Analysis of Tweets')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Number of Tweets')
plt.show()


# In[20]:


pip install transformers


# In[21]:


from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# prepare input
encoded_input = tokenizer(cleaned_tweets, return_tensors='pt')

# forward pass
output = model(**encoded_input)


# In[22]:


#a new dataframe 
results = pd.DataFrame(columns=['Text', 'length', 'spaces', 'char', 'verbs', 'start_i')
fd = nltk.FreqDist(final_tokens)
for tweet in cleaned_tweets:
    #Length of sentences                            
    text = tweet
    length = len(text)                            
    
    #length of words
    lengths = [len(w) for w in tweets ]
    lenfd = nltk.FreqDist(text)
    fd.tabulate()     
                                
    #spaces 
    spaces  = tweet.count(' ')
                            
# Print results
print(results)


# In[33]:


from nltk import CFG
grammar = CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | NP PP
VP -> V NP | VP PP
Det -> 'a' | 'the'
N -> 'dog' | 'cat'
V -> 'chased' | 'sat'
P -> 'on' | 'in'
""")


# In[35]:


parser = nltk.ChartParser(grammar)
trees = parser.nbest_parse(tweets)


# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


#Task6


# In[37]:


#a new dataframe 
results = pd.DataFrame(columns=['Text', 'Sentiment', 'Sentiment Score'])

#stemming+lemmmatization
for tweet in cleaned_tweets:
    text = tweet
    # Perform sentiment analysis using TextBlob
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    # Classify sentiment as positive, negative, or neutral
    if sentiment_score > 0:
        sentiment = 'positive'
    elif sentiment_score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    # Add results to DataFrame
    results = results.append({'Text': text, 'Sentiment': sentiment, 'Sentiment Score': sentiment_score}, ignore_index=True)

# Print results
print(results)


# In[57]:


from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
#preprocessed_text = [simple_preprocess(text) for text in results['1']]  # Assuming the text is in the first column
model1 = Word2Vec(results['Text'], min_count=1)


word2vec = pd.DataFrame(columns=['Text', 'model1'])


# In[58]:


word2vec


# In[70]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(tweets)
vectors


# In[72]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf  = TfidfVectorizer()
corpus = tfidf.fit_transform(tweets)
corpus


# In[ ]:





# In[ ]:





# In[73]:


#Task8


# In[74]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[78]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[80]:


pipeline = Pipeline([('CV', CountVectorizer()), 
                      ('Random Forest', RandomForestClassifier(n_estimators=50, criterion="entropy"))])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[81]:


from sklearn.naive_bayes import MultinomialNB


# In[83]:


pipeline = Pipeline([('CV', CountVectorizer()), 
                     ('MultinomialNB',MultinomialNB())])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[84]:


from sklearn.neural_network import MLPClassifier


# In[87]:


pipeline = Pipeline([('CV', CountVectorizer()), 
                     ('clf' ,MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1))])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[88]:


pipeline = Pipeline([('tfidf', TfidfVectorizer()), 
                      ('Random Forest', RandomForestClassifier(n_estimators=50, criterion="entropy"))])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[89]:


pipeline = Pipeline([('tfidf', TfidfVectorizer()), 
                      ('MultinomialNB',MultinomialNB())])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[90]:


pipeline = Pipeline([('tfidf', TfidfVectorizer()), 
                      ('clf' ,MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1))])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




