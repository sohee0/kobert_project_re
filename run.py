
# from flask import Flask, redirect, request, render_template, jsonify, url_for
# import model
# import tensorflow as tf

# app = Flask(__name__)

# print(__name__)


# @app.route('/')
# def home():
#      # model.predict_sentiment('기분이 너무 좋아')
#      return render_template('index.html')

# @app.route('/msg', methods=['POST'])
# def msg():
#      if request.method == 'POST':
#           msg = request.form
#      else:
#           msg = {}
     
#      return render_template('index.html', data=msg)

# @app.route('/predict', methods=['POST'])
# def predict():
#     return 'NEW_PAGE'



# if __name__ == '__main__': 
#      app.run(debug=True)
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

# @app.route('/msg', methods = ['POST', 'GET'])
# def msg():
#    if request.method == 'POST':
#       temp = request.form['userMsg']
#       # result에 들어 값으로 python처리!
#       result = '결과 > ' + temp
#       return render_template("index.html", result = result)
@app.route('/msg', methods = ['POST', 'GET'])
def msg():
   msg = request.form['userMsg']
   res={
      'name':'user',
      'msg':msg
   }
   return jsonify(res)
# @app.route('/first')
# def first():
#     return render_template('first.html')

if __name__ == '__main__':
   app.run(debug = True)