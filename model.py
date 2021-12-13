# Import Statements
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from statistics import mean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
nltk.download('punkt')


# read csv files for train data. 
def get_full_train_dataset():  
  # Set Seed
  np.random.seed(500)

  data_entertainment_train = pd.read_csv('./data_class_clean/data_entertainment_train.csv')
  data_politics_train = pd.read_csv('./data_class_clean/data_politics_train.csv')
  data_sports_train = pd.read_csv('./data_class_clean/data_sports_train.csv')
  data_tech_train = pd.read_csv('./data_class_clean/data_tech_train.csv')
  data_travel_train = pd.read_csv('./data_class_clean/data_travel_train.csv')
  data_finance_train = pd.read_csv('./data_class_clean/data_finance_train.csv')

  data_all_train=pd.concat([data_entertainment_train,data_sports_train,data_tech_train,data_travel_train,data_politics_train,data_finance_train],axis = 0)
  
  return data_all_train.dropna()

# read csv files for train data. 
def get_full_original_train_dataset():  
  # Set Seed
  np.random.seed(500)

  data_entertainment_train = pd.read_csv('./data_class_limited_data_train_test/data_entertainment_train.csv')
  data_politics_train = pd.read_csv('./data_class_limited_data_train_test/data_politics_train.csv')
  data_sports_train = pd.read_csv('./data_class_limited_data_train_test/data_sports_train.csv')
  data_tech_train = pd.read_csv('./data_class_limited_data_train_test/data_tech_train.csv')
  data_travel_train = pd.read_csv('./data_class_limited_data_train_test/data_travel_train.csv')
  data_finance_train = pd.read_csv('./data_class_limited_data_train_test/data_finance_train.csv')

  data_all_train=pd.concat([data_entertainment_train,data_sports_train,data_tech_train,data_travel_train,data_politics_train,data_finance_train],axis = 0)
  
  return data_all_train.dropna()

# read csv files for test data. 
def get_full_test_dataset():  
  # Set Seed
  np.random.seed(500)

  data_entertainment_test = pd.read_csv('./data_class_clean/data_entertainment_test.csv')
  data_politics_test = pd.read_csv('./data_class_clean/data_politics_test.csv')
  data_sports_test = pd.read_csv('./data_class_clean/data_sports_test.csv')
  data_tech_test = pd.read_csv('./data_class_clean/data_tech_test.csv')
  data_travel_test = pd.read_csv('./data_class_clean/data_travel_test.csv')
  data_finance_test = pd.read_csv('./data_class_clean/data_finance_test.csv')

  data_all_test=pd.concat([data_entertainment_test,data_sports_test,data_tech_test,data_travel_test,data_politics_test,data_finance_test],axis = 0)

  data_all_test['Class'].value_counts()
  return data_all_test.dropna()

# Get set sample data
def get_sample_dataset(data, count=None):
  
  labels_counts=data['Class'].value_counts()
  min_count= min(labels_counts)
  if count is not None:
    min_count = count
  # Divide by class
  data_0 = data[data['Class'] == 'Politics']
  data_1 = data[data['Class'] == 'Tech']
  data_2 = data[data['Class'] == 'Entertainment']
  data_3 = data[data['Class'] == 'Travel']
  data_4 = data[data['Class'] == 'Sports']
  data_5 = data[data['Class'] == 'Finance']

  data_0_under = data_0.sample(min_count)
  data_1_under = data_1.sample(min_count)
  data_2_under = data_2.sample(min_count)
  data_3_under = data_3.sample(min_count)
  data_4_under = data_4.sample(min_count)
  data_5_under = data_5.sample(min_count)

  data_sample = pd.concat([data_0_under, data_1_under,data_2_under,data_3_under,data_4_under,data_5_under], axis=0)

  data_sample.reset_index(inplace= True)
  data_sample = data_sample[['Class','Title','News']]
  return data_sample

# Prepare the dataset for training
def prepare_dataset_for_training(dataset,news_title):
  Encoder = LabelEncoder()
  Encoder.fit(dataset['Class'])
  Train_Y = Encoder.transform(dataset['Class'])

  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

  return  dataset[news_title], Train_Y, skf, Encoder.classes_, Encoder


# TF-IDF transformation for the data
def tf_idf_transformation(Train_X):

  Tfidf_vect = TfidfVectorizer()
  Tfidf_vect.fit(Train_X)

  return Tfidf_vect

##### Linear Regression Classifier
def fit_linear_regression(x, y, vectorization, skf):
  
  lr = LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
  lst_accu_stratified = []
  for train_index, test_index in skf.split(x, y):
      x_train_fold, x_test_fold = x[train_index], x[test_index]
      y_train_fold, y_test_fold = y[train_index], y[test_index]

      x_train_fold_vector = vectorization.transform(x_train_fold)
      x_test_fold_vector = vectorization.transform(x_test_fold)

      lr.fit(x_train_fold_vector, y_train_fold)
      y_pred = lr.predict(x_test_fold_vector)

      lst_accu_stratified.append(accuracy_score(y_test_fold, y_pred))
  
  return mean(lst_accu_stratified)*100

###### Naive Bayes Classifier
def fit_naive_bayes(x, y, vectorization, skf):
  
  nb = naive_bayes.MultinomialNB()
  lst_accu_stratified = []
  for train_index, test_index in skf.split(x, y):
      x_train_fold, x_test_fold = x[train_index], x[test_index]
      y_train_fold, y_test_fold = y[train_index], y[test_index]

      x_train_fold_vector = vectorization.transform(x_train_fold)
      x_test_fold_vector = vectorization.transform(x_test_fold)

      nb.fit(x_train_fold_vector, y_train_fold)
      y_pred = nb.predict(x_test_fold_vector)

      lst_accu_stratified.append(accuracy_score(y_test_fold, y_pred))
      
  return mean(lst_accu_stratified)*100

##### SVM Classifiier
def fit_svm(x, y, vectorization, skf):
  
  svm_model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
  lst_accu_stratified = []
  
  for train_index, test_index in skf.split(x, y):
      x_train_fold, x_test_fold = x[train_index], x[test_index]
      y_train_fold, y_test_fold = y[train_index], y[test_index]

      x_train_fold_vector = vectorization.transform(x_train_fold)
      x_test_fold_vector = vectorization.transform(x_test_fold)

      svm_model.fit(x_train_fold_vector, y_train_fold)
      y_pred = svm_model.predict(x_test_fold_vector)

      lst_accu_stratified.append(accuracy_score(y_test_fold, y_pred))
      
  return mean(lst_accu_stratified)*100

# Glove Embedding
def load_glove(word_index,max_features):
    EMBEDDING_FILE = './Model files/glove.6B.50d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:50]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Defining the Model
class BiLSTM(nn.Module):
  def __init__(self,max_features,embed_size,embedding_matrix):
    super(BiLSTM, self).__init__()
    self.hidden_size = 64
    drp = 0.1
    n_classes = 6
    self.embedding = nn.Embedding(max_features, embed_size)
    self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    self.embedding.weight.requires_grad = False
    self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
    self.linear = nn.Linear(self.hidden_size*4 , 64)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(drp)
    self.out = nn.Linear(64, n_classes)


  def forward(self, x):
    h_embedding = self.embedding(x)
    h_lstm, _ = self.lstm(h_embedding)
    avg_pool = torch.mean(h_lstm, 1)
    max_pool, _ = torch.max(h_lstm, 1)
    conc = torch.cat(( avg_pool, max_pool), 1)
    conc = self.relu(self.linear(conc))
    conc = self.dropout(conc)
    out = self.out(conc)
    return out

##### LSTM Classifiier

def fit_lstm(Train_X, Train_Y, skf):
  embed_size = 50 # how big is each word vector
  max_features = 40000 # how many unique words to use (i.e num rows in embedding vector)
  maxlen = 256 # max number of words in a question to use
  batch_size = 512 # how many samples to process at once
  accuracy = []
  for train_index, test_index in skf.split(Train_X, Train_Y): 
    n_epochs = 20
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(Train_X.iloc[train_index])
    
    train_X = tokenizer.texts_to_sequences(Train_X.iloc[train_index])
    test_X = tokenizer.texts_to_sequences(Train_X.iloc[test_index])
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    train_Y = Train_Y[train_index]
    test_Y = Train_Y[test_index]
    ## Pad the sentences 
    embedding_matrix = load_glove(tokenizer.word_index,max_features)

    model = BiLSTM(max_features,embed_size,embedding_matrix)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    if torch.cuda.is_available():
      model.cuda()


    # Load train and test in CUDA Memory
    if torch.cuda.is_available():
      x_train = torch.tensor(train_X, dtype=torch.long).cuda()
      y_train = torch.tensor(train_Y, dtype=torch.long).cuda()
      x_cv = torch.tensor(test_X, dtype=torch.long).cuda()
      y_cv = torch.tensor(test_Y, dtype=torch.long).cuda()
    else:
      x_train = torch.tensor(train_X, dtype=torch.long)
      y_train = torch.tensor(train_Y, dtype=torch.long)
      x_cv = torch.tensor(test_X, dtype=torch.long)
      y_cv = torch.tensor(test_Y, dtype=torch.long)
      

    # Create Torch datasets
    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_cv, y_cv)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    train_loss = []
    valid_loss = []

    for epoch in range(n_epochs):
      # Set model to train configuration
      model.train()
      avg_loss = 0.  
      for i, (x_batch, y_batch) in enumerate(train_loader):
        # Predict/Forward Pass
        y_pred = model(x_batch)
        # Compute loss
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
      
      # Set model to validation configuration -Doesn't get trained here
      model.eval()        
      avg_val_loss = 0.
      val_preds = np.zeros((len(x_cv),6))
      
      for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        val_preds[i * batch_size:(i+1) * batch_size] =F.softmax(y_pred).cpu().numpy()
      
      # Check Accuracy
      val_accuracy = sum(val_preds.argmax(axis=1)==test_Y)/len(test_index)
      train_loss.append(avg_loss)
      valid_loss.append(avg_val_loss)
      print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}'.format(epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy))
    accuracy.append(val_accuracy)
  return mean(accuracy)*100

def fit_test_linear_regression(Train_X, Train_Y, Test_X, Test_Y):
  # Linear Regression 
  lr = LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
  lr.fit(Train_X, Train_Y)
  y_pred = lr.predict(Test_X)
  print("LR Test Accuracy:",accuracy_score(Test_Y,y_pred)*100,"%")
  return y_pred

def fit_test_naive_bayes(Train_X, Train_Y, Test_X, Test_Y):
  # Naive Bayes 
  nb = naive_bayes.MultinomialNB()
  nb.fit(Train_X, Train_Y)
  y_pred = nb.predict(Test_X)
  print("NB Test Accuracy:",accuracy_score(Test_Y,y_pred)*100,"%")
  return y_pred

def fit_test_svm(Train_X, Train_Y, Test_X, Test_Y):
  # SVM
  svm_model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
  svm_model.fit(Train_X, Train_Y)
  y_pred = svm_model.predict(Test_X)
  print("SVM Test Accuracy:",accuracy_score(Test_Y,y_pred)*100,"%")
  return y_pred

def fit_test_lstm(Train_X, Train_Y, Test_X, Test_Y):
  embed_size = 50 # how big is each word vector
  max_features = 40000 # how many unique words to use (i.e num rows in embedding vector)
  maxlen = 256 # max number of words in a question to use
  batch_size = 512 # how many samples to process at once
  # for train_index, test_index in skf.split(Train_X, Train_Y): 
  n_epochs = 20
  tokenizer = Tokenizer(num_words=max_features)
  tokenizer.fit_on_texts(Train_X)
  train_X = tokenizer.texts_to_sequences(Train_X)
  test_X = tokenizer.texts_to_sequences(Test_X)

  train_X = pad_sequences(train_X, maxlen=maxlen)
  test_X = pad_sequences(test_X, maxlen=maxlen)

  train_Y = Train_Y
  test_Y = Test_Y
  ## Pad the sentences 
  embedding_matrix = load_glove(tokenizer.word_index,max_features)

  model = BiLSTM(max_features,embed_size,embedding_matrix)
  loss_fn = nn.CrossEntropyLoss(reduction='sum')
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
  if torch.cuda.is_available():
    model.cuda()


  # Load train and test in CUDA Memory
  if torch.cuda.is_available():
    x_train = torch.tensor(train_X, dtype=torch.long).cuda()
    y_train = torch.tensor(train_Y, dtype=torch.long).cuda()
    x_cv = torch.tensor(test_X, dtype=torch.long).cuda()
    y_cv = torch.tensor(test_Y, dtype=torch.long).cuda()
  else:
    x_train = torch.tensor(train_X, dtype=torch.long)
    y_train = torch.tensor(train_Y, dtype=torch.long)
    x_cv = torch.tensor(test_X, dtype=torch.long)
    y_cv = torch.tensor(test_Y, dtype=torch.long)
    

  # Create Torch datasets
  train = torch.utils.data.TensorDataset(x_train, y_train)
  valid = torch.utils.data.TensorDataset(x_cv, y_cv)

  # Create Data Loaders
  train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size)

  train_loss = []

  for epoch in range(n_epochs):
    # Set model to train configuration
    model.train()
    avg_loss = 0.  
    for i, (x_batch, y_batch) in enumerate(train_loader):
      # Predict/Forward Pass
      y_pred = model(x_batch)
      # Compute loss
      loss = loss_fn(y_pred, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avg_loss += loss.item() / len(train_loader)
      # Set model to validation configuration -Doesn't get trained here
    print('Epoch {}/{} \t loss={:.4f} \t '.format(epoch + 1, n_epochs, avg_loss))
  
  model.eval()        
  avg_val_loss = 0.
  val_preds = np.zeros((len(x_cv),6))
  
  for i, (x_batch, y_batch) in enumerate(test_loader):
    y_pred = model(x_batch).detach()
    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(test_loader)
    val_preds[i * batch_size:(i+1) * batch_size] =F.softmax(y_pred).cpu().numpy()
  
  # Check Accuracy
  val_accuracy = sum(val_preds.argmax(axis=1)==test_Y)/len(Test_Y)
  print('Test Accuracy',val_accuracy)
  return list(val_preds.argmax(axis=1))

if __name__ == '__main__':
  news_title = 'News'
  Train = get_full_train_dataset()
  Test = get_full_test_dataset()
  Test = get_sample_dataset(Test, count=100)
  Train = get_sample_dataset(Train, count=100)
  
  Train = Train[['Class','News']]
  print(Train['Class'].value_counts())
  print(len(Train))
  Train_X,Train_Y, skf, classes, Encoder = prepare_dataset_for_training(Train,'News')
  Test = Test[['Class','News']]
  Test_X = Test['News']
  Test_Y = Test['Class']
  Test_Y = Encoder.transform(Test_Y)

  y_pred=fit_test_lstm(Train_X, Train_Y, Test_X, Test_Y)
  print(y_pred)