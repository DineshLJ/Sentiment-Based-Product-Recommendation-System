from flask import Flask,render_template,url_for,request
from model import getProducts


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html',products=[] , page="home")

@app.route('/',methods=['POST'])
def recommend_product():
    return getProducts()

if __name__ == '__main__':
    app.run(debug=True )
