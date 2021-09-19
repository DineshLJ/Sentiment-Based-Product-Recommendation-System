
# import libraties
import pickle
from flask import Flask,render_template,url_for,request
import pandas as pd

def getProducts():
    try:
        user = request.form['username']
        mapping = pickle.load(open('./dataset_final.csv', 'rb'))
        user_rating = pickle.load(open('./user_final_rating.pkl', 'rb'))
        word_vectorizer = pickle.load(open('./Vectorizer.pkl', 'rb'))
        top_20_products = user_rating.loc[user].sort_values(ascending=False)[0:20]
        df = pd.merge(top_20_products, mapping, left_on='name', right_on='name', how='left')
        model = pickle.load(open('./final_model.pkl', 'rb'))
        reviews = df['reviews_text']
        reviews_transformed = word_vectorizer.transform(reviews.tolist())
        pred_val = model.predict(reviews_transformed)
        df['user_sentiment'] = pred_val
        df['user_sentiment'] = df['user_sentiment'].map({'Positive':0,'Negative':1})
        pro = df.groupby('name')['user_sentiment'].mean()
        pro = pro.reset_index()
        top_5_products = pro.sort_values(by='user_sentiment', ascending=False)[0:5]
        print(top_5_products)
        return render_template('results.html',products = top_5_products['name'] , page="result")
    except:
        return render_template('home.html', products=[] , page="result")
