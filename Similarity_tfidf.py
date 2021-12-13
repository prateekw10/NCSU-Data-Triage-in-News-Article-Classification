import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class Similarity_tfidf:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(documents)

    def document_similarity(self,s1,s2,similarity_process):
        if similarity_process == 'tfidf':
            score = self.process_tfidf_similarity(s1,s2)
        return score
  
    def process_tfidf_similarity(self,s1,s2):

    # To make uniformed vectors, both documents need to be combined first.
        s1=self.vectorizer.transform([s1])
        s2=self.vectorizer.transform([s2])

        score = cosine_similarity(s1, s2).flatten()
        return score[0]

def find_similarity_per_class(dataset, similarity_obj, method,cate):
    i = 0
    document_list = dataset.values.tolist()
    print(len(document_list))
    f = open('./Triage Data TfIdf Log Files/'+method+'_'+cate+'_'+'_log_file.txt', 'w')

    while i < len(document_list)-1:
        j = i+1
        while j < len(document_list):
            try:
                similarity = similarity_obj.document_similarity(document_list[i],document_list[j],method)
            except:
                similarity = 0.9
            if similarity > 0.7:
                f.write(str(document_list[i])+'\n'+str(document_list[j])+'\n\n')
                # print(str(document_list[i]),str(document_list[j]))
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
        method = 'tfidf'
        news_title = 'News'

        if cate == 'Entertainment':
            data_class_train = pd.read_csv('./data_class_clean/data_entertainment_train.csv')
        if cate == 'Sports':
            data_class_train = pd.read_csv('./data_class_clean/data_sports_train.csv')
        if cate == 'Tech':
            data_class_train = pd.read_csv('./data_class_clean/data_tech_train.csv')
        if cate == 'Travel':
            data_class_train = pd.read_csv('./data_class_clean/data_travel_train.csv')
        if  cate == 'Finance':
            data_class_train = pd.read_csv('./data_class_clean/data_finance_train.csv')
        if cate == 'Politics':
            data_class_train = pd.read_csv('./data_class_clean/data_politics_train.csv')

        similarity_obj = Similarity_tfidf(data_class_train[news_title])

        print('Similarity Method: ', method) 
        subset = pd.DataFrame(find_similarity_per_class(data_class_train[news_title], similarity_obj, method,cate), columns=[news_title])
        subset['Class'] = cate
        subset.to_csv('./Triage Data TfIdf/'+method+cate+news_title+'_Similarity_Dataset.csv',index = False)
        print(cate,len(subset))



if __name__ == '__main__':
    
    method = 'tfidf'
    cate = 'Politics'
    news_title = 'News'

    if cate == 'Entertainment':
        data_class_train = pd.read_csv('./data_class_clean/data_entertainment_train.csv')
    if cate == 'Sports':
        data_class_train = pd.read_csv('./data_class_clean/data_sports_train.csv')
    if cate == 'Tech':
        data_class_train = pd.read_csv('./data_class_clean/data_tech_train.csv')
    if cate == 'Travel':
        data_class_train = pd.read_csv('./data_class_clean/data_travel_train.csv')
    if  cate == 'Finance':
        data_class_train = pd.read_csv('./data_class_clean/data_finance_train.csv')
    if cate == 'Politics':
        data_class_train = pd.read_csv('./data_class_clean/data_politics_train.csv')

    similarity_obj = Similarity_tfidf(data_class_train[news_title])

    print('Similarity Method: ', method) 
    subset = pd.DataFrame(find_similarity_per_class(data_class_train[news_title], similarity_obj, method,cate), columns=[news_title])
    subset['Class'] = cate
    subset.to_csv('./Triage Data TfIdf/'+method+cate+news_title+'_Similarity_Dataset.csv',index = False)
    print(cate,len(subset))