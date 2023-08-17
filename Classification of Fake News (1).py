#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install nlp_utils')
get_ipython().system('pip install scikit-learn')


# In[3]:


import nlp_utils
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[26]:


df = pd.read_csv('FNC_dataset.csv')


# In[27]:


df.shape


# In[28]:





# In[29]:


pd.set_option('display.max_colwidth', -1)


# In[30]:


df['title']


# In[31]:


df['text']


# In[32]:


df['label'].value_counts()


# In[33]:


df.isnull().sum()


# In[34]:


df= df.dropna()


# In[35]:


df.isnull().sum()


# In[36]:


df.reset_index(inplace=True)


# In[37]:


df


# In[38]:


import re
import string


# In[39]:


alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
remove_n = lambda x: re.sub('\n', ' ', x)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ',x)


# In[40]:


df['text'] = df['text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)


# In[41]:


df['text']


# In[42]:


get_ipython().system('pip install nltk')


# In[43]:


import nltk
nltk.download('stopwords')


# In[51]:


df


# In[45]:


# for removing stopwords
from nltk.corpus import stopwords


# In[47]:


from nltk.stem.porter import PorterStemmer
import re
ps = PorterStemmer()
corpus = []


# In[48]:


for i in range(0, len(df)):
    review = re.sub('[^a-zA-z]', ' ', df['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[50]:


df


# In[52]:


Y = df['label']


# In[53]:


Y.head()


# In[54]:


X_train, X_test, Y_train, Y_test = train_test_split(df['text'], Y, test_size=0.30, random_state=40)


# In[57]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[58]:


tfidf_vect = TfidfVectorizer(stop_words = 'english', max_df=0.7)
tfidf_train = tfidf_vect.fit_transform(X_train)
tfidf_test = tfidf_vect.transform(X_test)


# In[59]:


print(tfidf_test)


# In[64]:


print(tfidf_vect.get_feature_names()[-10:])


# In[66]:


from sklearn.feature_extraction.text import CountVectorizer


# In[71]:


from sklearn.pipeline import Pipeline


# In[72]:


count_vect = CountVectorizer(stop_words = 'english')
count_train = count_vect.fit_transform(X_train.values)
count_test = count_vect.transform(X_test.values)


# In[73]:


#get the feature names of count vectorizer
print(count_vect.get_feature_names()[0:10])


# In[74]:


# data cleaning ends here and data modeling starts here
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[75]:


clf = MultinomialNB()
clf.fit(tfidf_train, Y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(Y_test, pred)
print("accuracy:  %0.3f" % score)
cm = metrics.confusion_matrix(Y_test, pred)


# In[76]:


print('worng prediction out of total')
print(Y_test != pred).sum(),'/',((Y_test == pred).sum()+(Y_test !=pred).sum())
print('Percentage accuracy: ', 100*accuracy_score(Y_test, pred))


# In[77]:


sns.heatmap(cm, cmap="plasma", annot=True)


# In[79]:


clf = MultinomialNB()
clf.fit(count_train, Y_train)
pred1 = clf.predict(count_test)
score = metrics.accuracy_score(Y_test, pred)
print("accuracy:  %0.3f" % score)
cm2 = metrics.confusion_matrix(Y_test, pred)
print(cm2)


# In[80]:


print('worng prediction out of total')
print(Y_test != pred1).sum(),'/',((Y_test == pred1).sum()+(Y_test !=pred1).sum())
print('Percentage accuracy: ', 100*accuracy_score(Y_test, pred1))


# In[81]:


sns.heatmap(cm2, cmap="plasma", annot=True)


# In[ ]:




