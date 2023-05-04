from flask import Flask,jsonify,url_for,render_template,request

app=Flask(__name__)

@app.route('/',methods=['GET'])
def calc():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import pickle
    raw_mail_data = pd.read_csv('mail_data1.csv')
    mail_data = raw_mail_data
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
    X = mail_data['Message']
    Y = mail_data['Category']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english')
    X_train_features = feature_extraction.fit_transform(X_train)
    input_mail=request.args['input']
    input_mail=[input_mail]
    input_data_features = feature_extraction.transform(input_mail)
    
    model=pickle.load(open('model.pkl','rb'))
    prediction = model.predict(input_data_features)
    if (prediction[0]==1):
        result = 'Not spam'
    else:
        result = 'Spam'

    return jsonify(prediction=result)
app.run(debug=True)
