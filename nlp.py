import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
items = pd.read_csv('./AmazonReviews/20191226-items.csv')
reviews = pd.read_csv('./AmazonReviews/20191226-reviews.csv')
print(items.head())
print(reviews.head())

df = pd.merge(reviews, items, how="left", left_on="asin", right_on="asin")
df['reviews'] = df['title_x'] + ' ' + df['body']

print(df['rating_x'])
# Add posivity label
df["positivity"] = df["rating_x"].apply(lambda x: 1 if x>3 else 0)
print(df['reviews'])
print(df['reviews'][0:1])
print(df.isnull().any())
df.dropna(inplace=True)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
c = []
for i in range(0,27071):
    reviews=re.sub('[^a-zA-Z]',' ',str(df['reviews'][i:i+1]))
    reviews=reviews.lower()
    reviews=reviews.split()
    reviews=[ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))]
    reviews=' '.join(reviews)
    c.append(reviews)

print(c[0:5])

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=6000)
x=cv.fit_transform(c).toarray()
y=df['positivity'].values
print(x[0:5])
print(x.shape)
print(y[0:5])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model= Sequential([
    Dense(3000,activation='sigmoid'),
    Dense(1000,activation='sigmoid'),
    Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=50,batch_size=10)

y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)

print(y_pred[0:5])
print(y_test[0:5])

print("Confusion matrix: ",confusion_matrix(y_test,y_pred))
print("Accuracy score: ",accuracy_score(y_test,y_pred))

model.save('sentimentAnalysis.h5')