import pandas as pd

class Similarity_Jaccard:
  def __init__(self):
    pass
  def document_similarity(self,s1,s2,similarity_process):
    if similarity_process == 'jaccard':
      score = self.process_jaccard_similarity(s1,s2)

    return score
  def trigrams(self,s):
    r = set()
    s = s.split(' ')
    for i in range(len(s) - 2):
        r.add(" ".join([t.lower() for t in s[i:i+3]]))
    return r
  
  def are_near_duplicates(self,s1, s2):
    return len(s1.intersection(s2)) / float(len(s1.union(s2)))
  
  def process_jaccard_similarity(self,s1,s2):
    return self.are_near_duplicates(self.trigrams(s1), self.trigrams(s2))




def find_similarity_per_class(dataset, similarity_obj, method,cate):
  i = 0
  document_list = dataset.values.tolist()
  f = open('./Triage Data Jaccard Log Files/'+method+'_'+cate+'_'+'_log_file.txt', 'w')

  while i < len(document_list)-1:
    j = i+1
    while j < len(document_list):
      try:
        similarity = similarity_obj.document_similarity(document_list[i],document_list[j],method)
      except:
        similarity = 0.9
      if similarity > 0.6:
        f.write(str(document_list[i])+'\n'+str(document_list[j])+'\n\n')
        print(str(document_list[i]),str(document_list[j]))
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
    method = 'jaccard'
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

    similarity_obj = Similarity_Jaccard()

    print('Similarity Method: ', method) 
    subset = pd.DataFrame(find_similarity_per_class(data_class_train[news_title], similarity_obj, method,cate), columns=[news_title])
    subset['Class'] = 'Politics'
    subset.to_csv(method+cate+news_title+'_Similarity_Dataset.csv',index = False)
    print(cate,len(subset))

    

if __name__ == '__main__':
    
  method = 'jaccard'
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

  similarity_obj = Similarity_Jaccard()

  print('Similarity Method: ', method) 
  subset = pd.DataFrame(find_similarity_per_class(data_class_train[news_title], similarity_obj, method,cate), columns=[news_title])
  subset['Class'] = 'Politics'
  subset.to_csv(method+cate+news_title+'_Similarity_Dataset.csv',index = False)
  print(cate,len(subset))
