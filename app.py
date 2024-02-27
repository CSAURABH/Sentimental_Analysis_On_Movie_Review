# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Naive Bayes model and TfidfVectorizer object from disk
filename = 'Movies_Review_Classification.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('count-Vectorizer.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html',prediction=[2])

# @app.route('/predict',methods=['POST'])
@app.route('/',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	print("message: " , message)
    	if len(message)==0:
    		return render_template('home.html', prediction=[2])
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	print('my_prediction : ', my_prediction)
    	return render_template('home.html', prediction=my_prediction)



if __name__ == '__main__':
	app.run(debug=False)