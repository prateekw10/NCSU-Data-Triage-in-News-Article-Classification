import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics.pairwise import cosine_similarity
import scipy

class Similarity_Glove:
  
    def __init__(self):
        self.gloveFile = './Model files/glove.6B.50d.txt'
        with open(self.gloveFile, encoding="utf8" ) as f:
            content = f.readlines()
        self.stopword_set = set(stopwords.words("english"))
        self.glove_model = {}
        for line in content:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            self.glove_model[word] = embedding
        print ("Done.",len(self.glove_model)," words loaded!")
    
    def document_similarity(self,s1,s2,similarity_process):
        if similarity_process == 'glove':
          score = self.process_glove_similarity(s1,s2)
        return score
    def removing_non_ascii(self, item):
        item =str(item)
        item = item.encode('ascii', 'ignore')
        item = item.decode()
        return item
    def preprocess(self,raw_text):

        # keep only words
        letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    
        # convert to lower case and split 
        words = letters_only_text.lower().split()
    
        # remove stopwords        
        cleaned_words = list(set([w for w in words if w not in self.stopword_set]))
        cleaned_words = list(set([w for w in cleaned_words if w in self.glove_model.keys()]))
        cleaned_words = list(set([self.removing_non_ascii(w) for w in cleaned_words]))
        return cleaned_words
    
    def process_glove_similarity(self,s1, s2):
        vector_1 = np.mean([self.glove_model[word] for word in self.preprocess(s1)],axis=0)
        vector_2 = np.mean([self.glove_model[word] for word in self.preprocess(s2)],axis=0)
        cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
        return 1-cosine


def find_similarity_per_class(dataset, similarity_obj, method,cate):
    i = 0
    document_list = dataset.values.tolist()
    f = open('./Triage Data Glove Log Files/'+method+'_'+cate+'_'+'_log_file.txt', 'w')

    while i < len(document_list)-1:
        j = i+1
        while j < len(document_list):
            similarity = similarity_obj.document_similarity(document_list[i],document_list[j],method)
            if similarity > .985:
                value=similarity_obj.removing_non_ascii(str(document_list[i])+'\n'+str(document_list[j])+'\n\n')
                f.write(value)
                del document_list[j]
            else:
                j+=1
        i+=1
        if i%10 == 0:
            print("i:",i)
      
    f.close()
    return document_list

def find_similarity():
    categories = ['Entertainment','Sport','Tech','Travel','Finance','Politics']
    for cate in categories:
        method = 'glove'
        similarity_obj = Similarity_Glove()
        news_title = 'News'

        if cate == 'Entertainment':
            data_class_train = pd.read_csv('.data_class_clean/data_entertainment_train.csv')
        if cate == 'Sports':
            data_class_train = pd.read_csv('.data_class_clean/data_sports_train.csv')
        if cate == 'Tech':
            data_class_train = pd.read_csv('.data_class_clean/data_tech_train.csv')
        if cate == 'Travel':
            data_class_train = pd.read_csv('.data_class_clean/data_travel_train.csv')
        if  cate == 'Finance':
            data_class_train = pd.read_csv('.data_class_clean/data_finance_train.csv')
        if cate == 'Politics':
            data_class_train = pd.read_csv('.data_class_clean/data_politics_train.csv')
        
        print('Similarity Method: ', method) 
        subset = pd.DataFrame(find_similarity_per_class(data_class_train[news_title], similarity_obj, method,cate), columns=[news_title])
        subset['Class'] = cate
        subset.to_csv('./Triage Data Glove/'+method+cate+news_title+'_Similarity_Dataset.csv',index = False)
        print(cate,len(subset))


if __name__ == '__main__':

    method = 'glove'
    cate = 'Tech'
    similarity_obj = Similarity_Glove()
    news_title = 'News'

    if cate == 'Entertainment':
        data_class_train = pd.read_csv('.data_class_clean/data_entertainment_train.csv')
    if cate == 'Sports':
        data_class_train = pd.read_csv('.data_class_clean/data_sports_train.csv')
    if cate == 'Tech':
        data_class_train = pd.read_csv('.data_class_clean/data_tech_train.csv')
    if cate == 'Travel':
        data_class_train = pd.read_csv('.data_class_clean/data_travel_train.csv')
    if  cate == 'Finance':
        data_class_train = pd.read_csv('.data_class_clean/data_finance_train.csv')
    if cate == 'Politics':
        data_class_train = pd.read_csv('.data_class_clean/data_politics_train.csv')
    
    print('Similarity Method: ', method) 
    subset = pd.DataFrame(find_similarity_per_class(data_class_train[news_title], similarity_obj, method,cate), columns=[news_title])
    subset['Class'] = cate
    subset.to_csv('./Triage Data Glove/'+method+cate+news_title+'_Similarity_Dataset.csv',index = False)
    print(cate,len(subset))
