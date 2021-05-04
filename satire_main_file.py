from flask import Flask, render_template, request, url_for, jsonify
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import tree
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import pymysql
import string
import json
import nltk
from nltk.corpus import stopwords

#connect to the database
password = 'Will@Power_1997'

# Connect to the database
connection = pymysql.connect(host='csmysql.cs.cf.ac.uk',
                             user='c1955887',
                             password=password,
                             db='c1955887_MED',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


def fetch_data(table):
    satireDic = {}
    l = []
    i = 0
    with connection.cursor() as cur:
        sql_select = f"SELECT * from {table} Order by sentiment_value "
        cur.execute(sql_select)
        records = cur.fetchall()
        cur.close()
        for row in records: 
            l = l + [row["phrase"].split()]
            
            if  i== 0 and row["sentiment_value"] == "satire":
                l.clear()
                i = 1
            satireDic[row["sentiment_value"]] = l

    return satireDic

def process_data(data):
    all_categories = []
    categories = []
    all_train_or_test_docs = []
    document_clean = []
    #config stopwords from nltk
    en_stops = set(stopwords.words('english'))

    for label in data:
        for document in data[label]:
            for word in document:
                if word not in en_stops:
                    document_clean.append(word.replace(",",""))
            processing = ' '.join(document_clean)
            document_clean.clear()
            processing = processing.strip(string.punctuation + string.digits)
            all_train_or_test_docs.append(processing)
            all_categories.append(label)
            if label not in categories:
                categories.append(label)

    #input("finish processing")
    return all_train_or_test_docs, all_categories, categories

def get_data_measure_table(type):
    if (type == "performance"):

        with connection.cursor() as cur:
            sql_select = "SELECT accuracy_score, precision_score, recall_score, f1_score from measurements_table Order by measure_id Desc Limit 1 "
            cur.execute(sql_select)
            records = cur.fetchall()
            cur.close()
    
        return round(records[0]['accuracy_score']*100,2), round(records[0]['precision_score']*100,2), round(records[0]['f1_score']*100,2), round(records[0]['recall_score']*100,2)
        
    if (type == "model"):
        with connection.cursor() as cur:
            sql_select = "SELECT model from classification_table  Order by classification_id Desc Limit 1"
            cur.execute(sql_select)
            records = cur.fetchall()
            cur.close()

        return records[0]['model']

    if (type == "vectorizer"):
        with connection.cursor() as cur:
            sql_select = "SELECT vectorizer from classification_table  Order by classification_id Desc Limit 1"
            cur.execute(sql_select)
            records = cur.fetchall()
            cur.close()

        return records[0]['vectorizer']




def insert_trained_values(list_rows):
   
    with connection.cursor() as cur:
        sql_insert = "INSERT INTO Feedback_table(sentiment_value,phrase) VALUES (%s,%s)"
        cur.executemany(sql_insert,list_rows)
        connection.commit()	
        cur.close()


def insert_measurements_values(accuracy_score,precision_score, recall_score, f1_score):

    with connection.cursor() as cur:
        sql_select = "Select classification_id from classification_table Order by classification_id Desc Limit 1"
        cur.execute(sql_select)
        records = cur.fetchall()
        cur.close()
        
    list_measurements = []
    # convert numpy.float64 to float
    row = (round(accuracy_score.item(),4), round(precision_score.item(),4), round(recall_score.item(),4), round(f1_score.item(),4), records[0]['classification_id'])
    list_measurements.append(row)
     
    with connection.cursor() as cur:
        sql_insert = "INSERT INTO measurements_table(accuracy_score,precision_score,recall_score,f1_score,classification_id) VALUES (%s,%s,%s,%s,%s)"
        cur.executemany(sql_insert,list_measurements)
        connection.commit()	
        cur.close()

def insert_classification_values(model,vectorizer):

    list_classification = []
    row = (model, vectorizer)
    list_classification.append(row)
     
    with connection.cursor() as cur:
        sql_insert = "INSERT INTO classification_table(model,vectorizer) VALUES (%s,%s)"
        cur.executemany(sql_insert,list_classification)
        connection.commit()	
        cur.close()



def train_model(docs,all_cmap):
    #Classification Models (Decision Tree / Logistic Regression)
    
    # 1. Decision Tree
    #clf = tree.DecisionTreeClassifier() 
    
    # 2. Logistic Regression
    clf = LogisticRegression(C=1)

    # Vectorizer (CountVectorizer / TfidfVectorizer)

    # 1. CountVectorizer 
    cvec = CountVectorizer(ngram_range=(1,2), analyzer='word')
  
    # 2. TfidfVectorizer
    #cvec = TfidfVectorizer()
  
    # vectorize the words
    cvec.fit(docs)
    X = cvec.transform(docs)
    y = np.array(all_cmap)
    

    numb_folds = 10
    kfold = StratifiedKFold(n_splits = numb_folds, shuffle=True, random_state=None)
    
    all_a = 0
    all_p = 0
    all_r = 0
    all_f = 0


    fold_counter = 1
    for train_index, test_index in kfold.split(X,y):

       
        X_train,X_test,y_train,y_test = X[train_index],X[test_index],y[train_index],y[test_index]
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        a = accuracy_score(y_test,preds)
        p = precision_score(y_test,preds)
        r = recall_score(y_test,preds)
        f1 = f1_score(y_test,preds)

        all_a += a
        all_p += p
        all_r += r
        all_f += f1
        fold_counter += 1

    print("accuracy: ",all_a/numb_folds) 
    print("precision: ",all_p/numb_folds) 
    print("recall: ",all_r/numb_folds)
    print("f1: ",all_f/numb_folds)  
    #save model and vectorizer
    save_model = pickle.dumps(clf)
    save_vectorizer = pickle.dumps(cvec)

    #insert classification values 
    insert_classification_values(save_model,save_vectorizer)
    #insert the the measurements in the database
    insert_measurements_values(all_a/numb_folds, all_p/numb_folds, all_r/numb_folds, all_f/numb_folds)
 

################################################################################################
#################################### 2 Part ####################################################
################################################################################################


#Variables
feedback_phrase = []
feedback_category = []
row_feedback = []


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def form():
	return render_template('index_1.html')

@app.route('/submitted', methods=['POST'])
def submitted_form():
    vectorizer = pickle.loads(get_data_measure_table("vectorizer"))
    model = pickle.loads(get_data_measure_table("model"))
    
    #get string from user
    text=request.form['input_text']
    vec=vectorizer.transform([text])
    #predict result
    pred=model.predict(vec)[0]
    pred=cmap[pred]
 
    return render_template('prediction_and_feedback_2.html',sentence=text,prediction=pred)

@app.route('/feedback_received', methods=['POST'])
def feedback_received():
    # feeback - correct or incorrect
    feedback = request.form['feedback']	
    sentence = request.form['sentence']
    pred = request.form['pred']
    right_answer = request.form['right_answer']		
    
    feedback_phrase.append(sentence)
    feedback_category.append(pred)
    # in case the prediction is wrong but the feedback seems valid
    if (right_answer == "satire" or right_answer == "non_satre") and feedback == "incorrect":
        feedback_category.pop()
        feedback_category.append(right_answer)
       
    # in case the prediction is wrong and the feedback seems invalid
    if right_answer == "none" and feedback =="incorrect":
        feedback_category.pop()
        feedback_phrase.pop()
       
    # in case the prediction is right or the prediction is wrong but with a valid feedback 
    if  feedback=="correct" or (feedback== "incorrect" and (right_answer=="satire" or right_answer=="non_satre") ):
        row = (feedback_category[len(feedback_category)-1], sentence)
        row_feedback.append(row)
        
        #insert trained data in the feedback database
        insert_trained_values(row_feedback)
        row_feedback.clear

       
 
    # if the user gives more than 4 feedbacks the model is trainned 
    if len(feedback_phrase) is 4:
        # feedback_category - value predicted by the system
        feedback_category.clear()
        feedback_phrase.clear()   
        all_data_1 = fetch_data("satire_table")
        all_data_2 = fetch_data("Feedback_table")
        #data from the 1st database - process
        docs_1, all_categories_1, cmap_1 = process_data(all_data_1)
        
        # data from the 2nd database - process
        docs_2, all_categories_2, cmap_2 = process_data(all_data_2)
        docs = docs_1 + docs_2
        all_cmap_1 = [0 for x in all_data_1['non_satre']]+[1 for x in all_data_1['satire']]
        all_cmap_2 = [0 for x in all_data_2['non_satre']]+[1 for x in all_data_2['satire']]
        all_data = all_cmap_1 + all_cmap_2
        
        #train the model
        train_model(docs,all_data)
        

    return render_template('feedback_results_3.html', sentence=sentence, feedback=feedback)

@app.route('/performance', methods=['GET'])
def performance():
    
    precision_res, accuracy_res, recall_res, f1_res = get_data_measure_table("performance")
    return render_template('performance_4.html', precision=precision_res, accuracy=accuracy_res, recall= recall_res, f1=f1_res)

if __name__ == '__main__':

     # Get data from database
    all_data = fetch_data("satire_table")
    # process the data to be used
    docs, all_categories, cmap = process_data(all_data)
    all_cmap = [0 for x in all_data['non_satre']]+[1 for x in all_data['satire']]
    # train the data before the application is run
    train_model(docs,all_cmap)

    # application is running
    app.run(host='0.0.0.0', port=8090, debug=True)
    