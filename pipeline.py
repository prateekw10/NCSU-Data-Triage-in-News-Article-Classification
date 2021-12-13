
import model as cm
import pandas as pd
import matplotlib.pyplot as plt
import Similarity_Jaccard as jd
import Similarity_tfidf as tf
import Similarity_Glove as gl
import Similarity_doc2vec as dc
from sklearn.metrics import confusion_matrix, accuracy_score
from random import random

def evaluateModels(Train_X, Train_Y, tf_idf_vect, skf):
   accuracy_measures = []

   # Linear Regression
   print("Fitting Linear Regression Model")
   lr_tfidf_accuracy = cm.fit_linear_regression(Train_X, Train_Y, tf_idf_vect, skf)
   accuracy_measures.append(lr_tfidf_accuracy)
   print("LR Accuracy:", lr_tfidf_accuracy)
   
   # Naive Bayes 
   print("Fitting Naive Bayes Model")
   nb_tfidf_accuracy = cm.fit_naive_bayes(Train_X, Train_Y, tf_idf_vect, skf)
   accuracy_measures.append(nb_tfidf_accuracy)
   print("NB Accuracy:", nb_tfidf_accuracy)
   
   # SVM
   print("Fitting SVM Model")
   svm_tfidf_accuracy = cm.fit_svm(Train_X, Train_Y, tf_idf_vect, skf)
   accuracy_measures.append(svm_tfidf_accuracy)
   print("SVM Accuracy:", svm_tfidf_accuracy)
   
   # LSTM
   print("Fitting LSTM Model")
   lstm_accuracy = cm.fit_lstm(Train_X, Train_Y, skf)
   accuracy_measures.append(lstm_accuracy)
   print("LSTM Accuracy:", lstm_accuracy)
   
  #  Plot accruacy graphs
  #  cm.plot_accuracy_graph(accuracy_measures) 
   
   print("Accuracy Measures: ",accuracy_measures)
   
def plot_confusion_matrix(cma, encoder_classes,model):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(cma)
  plt.title('Confusion matrix of the classifier')
  fig.colorbar(cax)
  ax.set_xticklabels([''] + encoder_classes, rotation=90)
  ax.set_yticklabels([''] + encoder_classes)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.savefig('Confusion matrix for '+model+str(random())+'.png')
   
def classification(Train_X, Train_Y, Test_X, Test_Y, encoder_classes, Encoder, tf_idf_vect):
  # Linear Regression
  y_pred = cm.fit_test_linear_regression(tf_idf_vect.transform(Train_X), Train_Y, tf_idf_vect.transform(Test_X), Test_Y)
  print("LR Test Accuracy:", accuracy_score(y_pred,Test_Y))
  cma = confusion_matrix(Encoder.inverse_transform(y_pred), Encoder.inverse_transform(Test_Y), labels=encoder_classes)
  plot_confusion_matrix(cma, encoder_classes,'Linear Regression')

  # Naive Bayes
  y_pred = cm.fit_test_naive_bayes(tf_idf_vect.transform(Train_X), Train_Y, tf_idf_vect.transform(Test_X), Test_Y)
  print("NB Test Accuracy:", accuracy_score(y_pred,Test_Y))
  cma = confusion_matrix(Encoder.inverse_transform(y_pred), Encoder.inverse_transform(Test_Y), labels=encoder_classes)
  plot_confusion_matrix(cma, encoder_classes,'Naive Bayes')

  # SVM
  y_pred = cm.fit_test_svm(tf_idf_vect.transform(Train_X), Train_Y, tf_idf_vect.transform(Test_X), Test_Y)
  print("SVM Test Accuracy:", accuracy_score(y_pred,Test_Y))
  cma = confusion_matrix(Encoder.inverse_transform(y_pred), Encoder.inverse_transform(Test_Y), labels=encoder_classes)
  plot_confusion_matrix(cma, encoder_classes,'SVM')

  # LSTM
  y_pred = cm.fit_test_lstm(Train_X, Train_Y, Test_X, Test_Y)
  print("LSTM Test Accuracy:", accuracy_score(y_pred,Test_Y))
  cma = confusion_matrix(Encoder.inverse_transform(y_pred), Encoder.inverse_transform(Test_Y), labels=encoder_classes)
  plot_confusion_matrix(cma, encoder_classes,'LSTM')
  
def fetchSubset(method):
  if method == 'jaccard':
    return pd.read_csv('./Data/Triage Data Jaccard/jaccardNews_Similarity_Dataset.csv')
  if method == 'tfidf':
    return pd.read_csv('./Data/Triage Data TfIdf/tfidfNews_Similarity_Dataset.csv')
  if method == 'glove':
    return pd.read_csv('./Data/Triage Data Glove/gloveNews_Similarity_Dataset.csv')
  if method == 'doc2vec':
    subset = pd.read_csv('./Data/Triage Data Doc2Vec/doc2vecNews_Similarity_Dataset.csv')
    return dc.cleaning_data(subset,news_title)
  
def similarity_detection():

  similarity_jaccard_obj = jd.Similarity_Jaccard()
  similarity_jaccard_obj.find_similarity()
  
  similarity_tfidf_obj = tf.Similarity_tfidf()
  similarity_tfidf_obj.find_similarity()
  
  similarity_glove_obj = gl.Similarity_Glove()
  similarity_glove_obj.find_similarity()
  
  similarity_doc2vec_obj = dc.Similarity_doc2vec()
  similarity_doc2vec_obj.find_similarity()

    
if __name__ == '__main__':
  
  ####### Data Preparation
  
  news_title = 'News'
  sample_count = None
  
  data_all_train = cm.get_full_train_dataset()
  data_all_train = cm.get_sample_dataset(data_all_train, sample_count)
  
  print("Training Data Loaded")
  
  test_data = cm.get_full_test_dataset()
  test_data = cm.get_sample_dataset(test_data, sample_count)
  
  print("Testing Data Loaded")
  
  Train_X, Train_Y, skf, encoder_classes, Encoder = cm.prepare_dataset_for_training(data_all_train, news_title)
  
  # Tf-Idf Transformation
  tf_idf_vect = cm.tf_idf_transformation(Train_X)
  
  Test_X, Test_Y = test_data[news_title], Encoder.transform(test_data['Class'])
  
  print("Data Processing Complete")
  
  ####### Pre-Triage Classificaion Pipeline
  
  evaluateModels(Train_X, Train_Y, tf_idf_vect, skf)
  
  classification(Train_X, Train_Y, Test_X, Test_Y, encoder_classes, Encoder, tf_idf_vect)
  
  ####### Document Similarity Pipeline
  
  similarity_detection()
  
  ####### Post-Triage Classificaion Pipeline
  
  print("------ Post Triage ----------")
  
  methods = ['tfidf'] #'glove','doc2vec'
  
  for method in methods:
    
      print("Evaluting " + method.upper() + " Similiarity Dataset")
     
      subset = fetchSubset(method)
     
      Train_X, Train_Y, skf, encoder_classes, Encoder = cm.prepare_dataset_for_training(subset,news_title)
     
      tf_idf_vect = cm.tf_idf_transformation(Train_X)
     
      Test_X, Test_Y = test_data[news_title], Encoder.transform(test_data['Class'])
  
      evaluateModels(Train_X, Train_Y, tf_idf_vect, skf)
     
      classification(Train_X, Train_Y, Test_X, Test_Y, encoder_classes, Encoder, tf_idf_vect)
    