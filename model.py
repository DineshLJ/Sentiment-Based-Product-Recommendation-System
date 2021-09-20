
# import libraties
import pickle
from flask import Flask,render_template,url_for,request
import pandas as pd

def getProducts():
    try:
        username = username.strip()
        user = request.form['username']
        mapping = pickle.load(open('./pickle/dataset_final.csv', 'rb'))
        user_rating = pickle.load(open('./pickle/user_final_rating.pkl', 'rb'))
        word_vectorizer = pickle.load(open('./pickle/Vectorizer.pkl', 'rb'))
        top_20_products = user_rating.loc[user].sort_values(ascending=False)[0:20]
        df = pd.merge(top_20_products, mapping, left_on='name', right_on='name', how='left')
        model = pickle.load(open('./pickle/final_model.pkl', 'rb'))
        reviews = df['reviews_text'].values.astype('U')
        reviews_transformed = word_vectorizer.transform(reviews.tolist())
        pred_val = model.predict(reviews_transformed)
        df['user_sentiment'] = pred_val
        pro = df.groupby('name')['user_sentiment'].mean()
        pro = pro.reset_index()
        top_5_products = pro.sort_values(by='user_sentiment', ascending=False)[0:5]
        print(top_5_products)
        return render_template('results.html',products = top_5_products['name'] , page="result")
    except:
        return render_template('results.html', products=[] , page="result")
