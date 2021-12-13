import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class Similarity_doc2vec:
    def __init__(self, documents):
        self.lemmatizer = WordNetLemmatizer()
        self.model_Doc2Vec = Doc2Vec.load('./Model files/doc2vec.bin')
        self.stop_words = set(stopwords.words('english'))
        self.punctuations = string.punctuation

    def document_similarity(self,s1,s2,similarity_process):
        if similarity_process == 'doc2vec':
          score = self.process_doc2vec_similarity(s1,s2)
        return score
  
    def preprocess(self,text):
    # Steps:
    # 1. lowercase
    # 2. Lammetize. (It does not stem. Try to preserve structure not to overwrap with potential acronym).
    # 3. Remove stop words.
    # 4. Remove punctuations.
    # 5. Remove character with the length size of 1.

        lowered = str.lower(text)
    
        
        word_tokens = word_tokenize(lowered)
    
        words = []
        for w in word_tokens:
            w=self.removing_non_ascii(w)
            if w not in self.stop_words:
                if w not in self.punctuations:
                    if len(w) > 1:
                        lemmatized = self.lemmatizer.lemmatize(w)
                        words.append(lemmatized)
    
        return words

    def process_doc2vec_similarity(self,s1,s2):

        s1 = self.preprocess(s1)
        s1 = list(filter(lambda x: x in self.model_Doc2Vec.wv.vocab.keys(), s1))
        s1 = self.model_Doc2Vec.infer_vector(s1)
    
    
        s2 = self.preprocess(s2)
        s2 = list(filter(lambda x: x in self.model_Doc2Vec.wv.vocab.keys(), s2))
        s2 = self.model_Doc2Vec.infer_vector(s2)
    
        score = cosine_similarity([s1], [s2]).flatten()
    
        return score[0]
    def removing_non_ascii(self, item):
        item =str(item)
        item = item.encode('ascii', 'ignore')
        item = item.decode()
        return item


def find_similarity_per_class( dataset, similarity_obj, method,cate):
    i = 0
    document_list = dataset.values.tolist()
    f = open('./Triage Data Doc2Vec Log Files/'+ method+'_'+cate+'_'+'_log_file.txt', 'w')
    
    while i < len(document_list)-1:
        j = i+1
        while j < len(document_list):
            try:
                similarity = similarity_obj.document_similarity(document_list[i],document_list[j],method)
            except:
                similarity = 0.9
            if similarity > 0.7:
                value=similarity_obj.removing_non_ascii(str(document_list[i])+'\n'+str(document_list[j])+'\n\n')
                f.write(value)
                del document_list[j]
            else:
                j+=1
        i+=1
        if i%10 == 0:
            print("i:",i)
            print("subset size:",len(document_list))
    
    f.close()
    return document_list


def find_similarity():
    categories = ['Entertainment','Sport','Tech','Travel','Finance','Politics']
    for cate in categories:
        method = 'doc2vec'
        news_title = 'News'
        print(method,cate,news_title)
        if cate == 'Entertainment':
            data_class_train = pd.read_csv('data_class_limited_data_train_test/data_entertainment_train.csv')
        if cate == 'Sports':
            data_class_train = pd.read_csv('data_class_limited_data_train_test/data_sports_train.csv')
        if cate == 'Tech':
            data_class_train = pd.read_csv('data_class_limited_data_train_test/data_tech_train.csv')
        if cate == 'Travel':
            data_class_train = pd.read_csv('data_class_limited_data_train_test/data_travel_train.csv')
        if cate == 'Finance':
            data_class_train = pd.read_csv('data_class_limited_data_train_test/data_finance_train.csv')
        if cate == 'Politics':
            data_class_train = pd.read_csv('data_class_limited_data_train_test/data_politics_train.csv')


        similarity_obj = Similarity_doc2vec(data_class_train[news_title])


        print('Similarity Method: ', method) 
        subset = pd.DataFrame(similarity_obj.find_similarity_per_class(data_class_train[news_title], similarity_obj, method,cate), columns=[news_title])
        subset['Class'] = cate
        subset.to_csv('./Triage Data Doc2Vec/'+method+cate+news_title+'_Similarity_Dataset.csv',index = False)
        print(cate,len(subset))



if __name__ == '__main__':

    method = 'doc2vec'
    cate = 'Politics'
    news_title = 'News'
    print(method,cate,news_title)
    if cate == 'Entertainment':
        data_class_train = pd.read_csv('data_class_limited_data_train_test/data_entertainment_train.csv')
    if cate == 'Sports':
        data_class_train = pd.read_csv('data_class_limited_data_train_test/data_sports_train.csv')
    if cate == 'Tech':
        data_class_train = pd.read_csv('data_class_limited_data_train_test/data_tech_train.csv')
    if cate == 'Travel':
        data_class_train = pd.read_csv('data_class_limited_data_train_test/data_travel_train.csv')
    if cate == 'Finance':
        data_class_train = pd.read_csv('data_class_limited_data_train_test/data_finance_train.csv')
    if cate == 'Politics':
        data_class_train = pd.read_csv('data_class_limited_data_train_test/data_politics_train.csv')


    similarity_obj = Similarity_doc2vec(data_class_train[news_title])


    print('Similarity Method: ', method) 
    subset = pd.DataFrame(similarity_obj.find_similarity_per_class(data_class_train[news_title], similarity_obj, method,cate), columns=[news_title])
    subset['Class'] = cate
    subset.to_csv('./Triage Data Doc2Vec/'+method+cate+news_title+'_Similarity_Dataset.csv',index = False)
    print(cate,len(subset))