import os
import datacleaning as dc
import pandas as pd


data_entertainment=pd.read_csv('./data_class/data_entertainment.csv')
data_politics=pd.read_csv('./data_class/data_politics.csv')
data_sports=pd.read_csv('./data_class/data_sport.csv')
data_tech=pd.read_csv('./data_class/data_tech.csv')
data_travel=pd.read_csv('./data_class/data_travel.csv')
data_finance=pd.read_csv('./data_class/data_finance.csv')


data_all= {
    'data_entertainment':data_entertainment,
    'data_tech':data_tech,
    'data_sports' : data_sports,
    'data_travel' : data_travel,
    'data_politics':data_politics,
    'data_finance':data_finance
    }

for x in data_all:
    print(data_all[x]['Class'].value_counts())

data_all = dc.dropna_df(data_all)
# dc.plot_before(data_all)

for keys in data_all:
    data_all[keys]['News'] = data_all[keys][data_all[keys]['News'].str.count(' ') >= 128]['News']
    data_all[keys]['Title'] = data_all[keys][data_all[keys]['Title'].str.count(' ') >= 4]['Title']

dc.data_all = dc.dropna_df(data_all)
# dc.plot_before(data_all)

min = dc.min_count(data_all)
data_all['data_politics']=data_all['data_politics'][:min]
data_all['data_sports']=data_all['data_sports'][:min]
data_all['data_travel']=data_all['data_travel'][:min]
data_all['data_tech']=data_all['data_tech'][:min]
data_all['data_finance']=data_all['data_finance'][:min]
data_all['data_entertainment']=data_all['data_entertainment'][:min]

data_all['data_entertainment'].to_csv('./data_class_limited_data/data_entertainment.csv',index=False)
data_all['data_tech'].to_csv('./data_class_limited_data/data_tech.csv',index=False)
data_all['data_sports'].to_csv('./data_class_limited_data/data_sport.csv',index=False)
data_all['data_travel'].to_csv('./data_class_limited_data/data_travel.csv',index=False)
data_all['data_politics'].to_csv('./data_class_limited_data/data_politics.csv',index=False)
data_all['data_finance'].to_csv('./data_class_limited_data/data_finance.csv',index=False)

data_entertainment=pd.read_csv('./data_class_limited_data/data_entertainment.csv')
data_politics=pd.read_csv('./data_class_limited_data/data_politics.csv')
data_sports=pd.read_csv('./data_class_limited_data/data_sport.csv')
data_tech=pd.read_csv('./data_class_limited_data/data_tech.csv')
data_travel=pd.read_csv('./data_class_limited_data/data_travel.csv')
data_finance=pd.read_csv('./data_class_limited_data/data_finance.csv')

data_all= {
    'data_entertainment':   data_entertainment,
    'data_tech':    data_tech,
    'data_sports' : data_sports,
    'data_travel' : data_travel,
    'data_politics':data_politics,
    'data_finance':data_finance
    }
data_all_split={}


for keys in data_all:
    data_all[keys] = data_all[keys].sample(frac=1)
    train_size = 0.8
    train_end = int(len(data_all[keys])*train_size)
    data_all_split[keys+'_train'] = data_all[keys][:train_end]
    data_all_split[keys+'_test'] = data_all[keys][train_end:]


for keys in data_all_split:
  data_all_split[keys].to_csv('./data_class_limited_data_train_test/'+keys+'.csv',index=False)

data_all = data_all_split 

data_all = dc.dropna_df(data_all)
# dc.plot(data_all)


# removing punctuations from title and news.replace('[^\w\s]','')
for keys in data_all:
    data_all[keys]['Title'] = data_all[keys]['Title'].str.lower().apply(dc.puncuation)
    data_all[keys]['News'] = data_all[keys]['News'].str.lower().apply(dc.puncuation)
data_all = dc.dropna_df(data_all)
# dc.plot(data_all)

# removing Stopwords
stop = dc.stop_words()
for keys in data_all:
    data_all[keys]['Title'] = data_all[keys]['Title'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
    data_all[keys]['News'] = data_all[keys]['News'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
data_all = dc.dropna_df(data_all)
# dc.plot(data_all)

# removing numeric characters from title and news
for keys in data_all:
    data_all[keys]['Title'] = data_all[keys]['Title'].str.replace('\d+', '')
    data_all[keys]['News'] = data_all[keys]['News'].str.replace('\d+', '')
data_all = dc.dropna_df(data_all)
# dc.plot(data_all)

# Removing non ascii
for keys in data_all:
    data_all[keys]['Title'] = data_all[keys]['Title'].apply(dc.removing_non_ascii)
    data_all[keys]['News'] = data_all[keys]['News'].apply(dc.removing_non_ascii)
data_all = dc.dropna_df(data_all)
# dc.plot(data_all)


# Lemmatization
w_tokenizer,lemmatizer = dc.lemmatize()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

for keys in data_all:
    data_all[keys]['Title'] = data_all[keys]['Title'].apply(lemmatize_text)
    data_all[keys]['News'] = data_all[keys]['News'].apply(lemmatize_text)
data_all = dc.dropna_df(data_all)
# dc.plot(data_all)

# Join String
for keys in data_all:
    data_all[keys]['Title'] = data_all[keys]['Title'].apply(dc.string_join)
    data_all[keys]['News'] = data_all[keys]['News'].apply(dc.string_join)
data_all = dc.dropna_df(data_all)
# dc.plot(data_all)

# Deleting news articles less than 64 word
for keys in data_all:
    data_all[keys]['News'] = data_all[keys][data_all[keys]['News'].str.count(' ') >= 64]['News']
    data_all[keys]['Title'] = data_all[keys][data_all[keys]['Title'].str.count(' ') >= 4]['Title']
data_all = dc.dropna_df(data_all)
# dc.plot(data_all)

# Removing values if they became Nan
data_all = dc.dropna_df(data_all)
# dc.plot(data_all)

# To Trim the length of the News Article
for keys in data_all:
    data_all[keys]['News'] = data_all[keys]['News'].apply(dc.length_of_news)
data_all = dc.dropna_df(data_all)
# dc.plot(data_all)

# To Trim the length of the News Title
for keys in data_all:
    data_all[keys]['Title'] = data_all[keys]['Title'].apply(dc.length_of_title)
data_all = dc.dropna_df(data_all)

# Visulization 
dc.length_plot(data_all)
# dc.plot(data_all)

#saving the model
for keys in data_all_split:
  data_all_split[keys].to_csv('./data_class_clean/'+keys+'.csv',index=False)

for keys in data_all_split:
  print(data_all_split[keys]['Class'].value_counts())