from flask import Flask,render_template,request
import pickle
import warnings 
warnings.filterwarnings('ignore')
# import requests
from sklearn.metrics import f1_score
import gspread
from sklearn.metrics import recall_score
# import json
from oauth2client.service_account import ServiceAccountCredentials
app = Flask(__name__)
trained_model = None
complaint='' 
summary=''
predicted_category=''
response=''


def initialize_model():
    global trained_model

    import pandas as pd
    import pickle
    
    
    data=pd.read_csv('.\Email_train.csv')
    
    data_1=pd.DataFrame()
    
    data_1['complaint_text']=data.complaint_text
    data_1['Category_of_review']=data.Category_of_review
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    
    #checking for null values and dropping null values
    data_1.isnull().sum()
    
    data_1.dropna(inplace=True)
    
    """## Model Traning & Multinomial Naive Bayes"""
    
    X_train, X_test, y_train, y_test = train_test_split(data_1['complaint_text'], data_1['Category_of_review'], test_size=0.2, random_state=42)
    
    # Convert text data to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    classifier_NB = MultinomialNB()
    classifier_NB.fit(X_train_tfidf, y_train)
    
    # Make predictions on the test data
    predictions_NB = classifier_NB.predict(X_test_tfidf)
    
    # Calculate accuracy
    accuracy_NB = accuracy_score(y_test, predictions_NB)
    print("Accuracy using Naive Bayes :",accuracy_NB*100)
    #calcuating precision
    precision_NB=precision_score(y_test, predictions_NB, pos_label='positive',average='micro')
    print("precision score using Naive bayes: ",precision_NB*100)
    """## Using Random Forest"""
    
    from sklearn.ensemble import RandomForestClassifier
    classifier_RF = RandomForestClassifier(n_estimators=56, random_state=42)  # You can adjust the number of trees (n_estimators) as needed
    classifier_RF.fit(X_train_tfidf, y_train)
    
    # Make predictions on the test data
    predictions_RF = classifier_RF.predict(X_test_tfidf)
    
    # Calculate accuracy
    accuracy_Random_Forest = accuracy_score(y_test, predictions_RF)
    print("Accuracy using Random Forest :",accuracy_Random_Forest*100)
    #calcuating precision
    precision_RF=precision_score(y_test, predictions_RF, pos_label='positive',average='micro')
    print("precision score using random forest: ",precision_RF*100)
    
    """## Using SVM"""
    from sklearn.svm import SVC
    svm_classifier = SVC(kernel='linear', C=1.0)  # You can adjust kernel and C parameter as needed
    svm_classifier.fit(X_train_tfidf, y_train)
    
    # Make predictions on the test data
    predictions_svm = svm_classifier.predict(X_test_tfidf)
    
    # Calculate accuracy
    accuracy_svm = accuracy_score(y_test, predictions_svm)
    print("Accuracy using SVM:", accuracy_svm * 100)
    #calcuating precision
    precision_svm=precision_score(y_test, predictions_svm, pos_label='positive',average='micro')
    print("precision score using svm: ",precision_svm*100)
    f1_score_svm=f1_score(y_test,predictions_svm,pos_label='positive',average='micro')
    print("F1_score using SVM: ",f1_score_svm*100)
    recall_score_svm=recall_score(y_test,predictions_svm,pos_label='positive',average='micro')
    print("recall using svm: ",recall_score_svm*100)
    
    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(svm_classifier, model_file)
    with open('vectorizer.pkl', 'wb') as vector_file:
        pickle.dump(vectorizer, vector_file)

# Route to initialize the model and generate the pic


initialize_model()
print("Model initialized and pickle file generated.")

@app.route('/',methods=['GET'])
def home():
    return render_template("homepage.html")
@app.route('/post', methods = ['GET','POST'])
def gfg():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       first_name = request.form.get("fname")
       # getting input with name = lname in HTML form 
       last_name = request.form.get("lname") 
       print(first_name)
       return "Your name is "+first_name + last_name
    
    return "hello world"

@app.route('/single_product2')
def product_page2():
    return render_template('single_product2.html')
@app.route('/single_product')
def product_page():
    return render_template('single_product.html')
@app.route('/single_product3')
def product_page3():
    return render_template('single_product3.html')
@app.route('/single_product4')
def product_page4():
    return render_template('single_product4.html')
@app.route('/submit_complaint')
def submit_complaint():
    return render_template('complaint_form.html')
@app.route('/homepage')
def homepage():
    return render_template('homepage.html')



@app.route('/run_script', methods=['GET','POST'])
def run_script():

    if request.method == 'POST':
       complaint = request.form["complaint"]
      #  print(complaint)
    print(complaint,'is')
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name('.\json_keys.json', scope)
    client = gspread.authorize(credentials)
    gsheet = client.open("Customer_Complaints_Data").sheet1
    non_empty_rows = gsheet.get_all_values()
    last_row_number = len(non_empty_rows)
    # print(last_row_number)
    
    import json
    import spacy
    # from spacy.lang.en import English
    from spacy.lang.en.stop_words import STOP_WORDS
    from string import punctuation
    from heapq import nlargest
    

    stopwords = list(STOP_WORDS)

    nlp = spacy.load('en_core_web_sm')
    
    doc=nlp(complaint)
    punctuation = punctuation + '\n'
    word_frequencies = {}
    for word in doc:
     if word.text.lower() not in stopwords:
       if word.text.lower() not in punctuation:
         if word.text not in word_frequencies.keys():
           word_frequencies[word.text] = 1
         else:
           word_frequencies[word.text] += 1
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
      for word in sent:
        if word.text.lower() in word_frequencies.keys():
          if sent not in sentence_scores.keys():
            sentence_scores[sent] = word_frequencies[word.text.lower()]
          else:
            sentence_scores[sent] += word_frequencies[word.text.lower()]
    select_length = int(len(sentence_tokens)*0.85)
    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    model = pickle.load(open("trained_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    
    new_text_tfidf = vectorizer.transform([complaint])
    predicted_category = model.predict(new_text_tfidf)[0]
    print(predicted_category)
    response=''
    i=predicted_category.lower()
    if(i=='product quality'):
       response="Thank you for sharing your feedback with us. We're truly sorry to hear about your experience with our product. We take quality issues seriously and will investigate this matter further. Your input helps us improve, and we appreciate your patience and understanding. Please reach out to our customer support for assistance with returns or replacements."
    elif(i=='fake advertisement'):
       response="We apologize for any inconvenience you've experienced due to the advertisements. Our intention is to provide accurate information, and we take this matter seriously. We will investigate and rectify any misleading content. Please contact our customer support team for any specific issues or concerns. Your feedback is invaluable in helping us improve."
    elif(i=='shipping' or i=='shipping problem'):
       response="We apologize for the shipping difficulties you've faced. We understand the importance of timely delivery and are actively working to improve our shipping process. Please contact our customer support, and we will do our best to resolve any outstanding issues or concerns. Your feedback helps us enhance our services. Thank you for your patience."
    elif(i=='warranty'or i=='warranty card'):
       response="We apologize for any issues you've encountered with our warranty cards. Your feedback is important to us, and we're committed to ensuring a smooth warranty process. Please reach out to our customer support, and we'll be happy to assist with any concerns or questions you may have regarding your warranty. Thank you for bringing this to our attention."
    else:
       if(i=='positive review'):
          response="Thank you for taking the time to leave us a review. We are thrilled to hear that you loved your experience with us. Your kind words mean a lot to our team!"
    print(response)
    values_to_update = [
    [summary, predicted_category, response]]
    start_column_letter = chr(ord('A') + 3)
    end_column_letter = chr(ord('A') + 5)
    
    try:
      #  cell_range = f"{gsheet.get_addr_int(last_row_number,4)}:{gsheet.get_addr_int(last_row_number, 6)}"
       cell_range = f"{start_column_letter}{last_row_number}:{end_column_letter}{last_row_number}"

       gsheet.update(range_name=cell_range, values=values_to_update)
    except Exception as e:
       print("An error occurred:", e)
   
    return "data sent successfully"

if __name__ == '__main__':
    app.run(debug=True)
