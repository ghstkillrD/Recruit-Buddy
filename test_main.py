#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[14]:


# Load job listings data
job_data = pd.read_csv('job_data.csv')

# Preprocess data
job_data.drop_duplicates(inplace=True)
job_data.dropna(inplace=True)

# Define features
features = ['Job Salary', 'Job Experience', 'Key Skills', 'Role Category', 'Functional Area', 'Industry', 'Job Title']


# In[15]:


# Vectorize features
vectorizer = TfidfVectorizer(stop_words='english')
job_features = vectorizer.fit_transform(job_data[features].apply(lambda x: ' '.join(x), axis=1))
user_features = vectorizer.transform(['data analyst'])


# In[16]:


# Train machine learning model
X_train, X_test, y_train, y_test = train_test_split(job_features, job_data['Role Category'], test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Get job recommendations for user
user_similarity = cosine_similarity(user_features, job_features).flatten()
job_data['similarity'] = user_similarity
job_data = job_data.sort_values('similarity', ascending=False)
recommendations = job_data.head(10)['Job Title'].tolist()
print('Recommended jobs:', recommendations)


# In[ ]:




