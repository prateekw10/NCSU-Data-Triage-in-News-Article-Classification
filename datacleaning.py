import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
nltk.download('wordnet')
nltk.download('stopwords')

# Functions
def plot_before(data_all):
  length = []
  for data in data_all:
      length.append(data_all[data].shape[0])
      print(data,data_all[data].shape[0]),
  
  plt.bar(data_all.keys(),length)
  plt.xlabel("News Categories")
  plt.ylabel("Number of Instances")
  plt.title("Cleaned Dataset")
  plt.xticks(list(data_all.keys()),['Entertainment','Technology','Sport','Travel','Politics','Finance'],rotation='vertical')
  plt.show()

def plot(data_all):
  length = []
  for data in data_all:
      length.append(data_all[data].shape[0])
      print(data,data_all[data].shape[0]),
  
  plt.bar(data_all.keys(),length)
  plt.xlabel("News Categories")
  plt.ylabel("Number of Instances")
  plt.title("Cleaned Dataset")
  plt.xticks(list(data_all.keys()),['Entertainment Train','Entertainment Test','Technology Train','Technology Test','Sport Train','Sport Test','Travel Train','Travel Test','Politics Train','Politics Test','Finance Train', 'Finance Test'],rotation='vertical')
  plt.show()

def length_of_news(item):
    item = str(item)
    if len(item.split(' ')) > 512 :
        item= ' '.join(item.split(' ')[:512])
    return item


def length_of_title(item):
    item = str(item)
    if len(item.split(' ')) > 20 :
        item= ' '.join(item.split(' ')[:20])
    return item

# Defining Funciton to remove puncuations
def puncuation(item):
    item=str(item)
    for pun in [',', "'",'.', '"', '-', ':', ";", "/", "{", "}", '[', ']', '#', '%', '&', '*', '(', ')', '@','’', '!', '~', '`', '$', '?', '^', '_', \
                '=', '-','|','‘',',', '.', '"', ':', ')','(', '!', '?', '|', ';', "'", '$', '&','/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', \
                '•',  '~', '@', '£','·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›','♥', '←', '×', '§', '″', '′', 'Â', '█', '½',\
                'à', '…', '“', '★', '”','–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾','═', '¦', '║', '―', '¥', '▓', '—', '‹', \
                '─', '▒', '：', '¼', '⊕', '▼','▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲','è', '¸', '¾', 'Ã', '⋅', '‘', '∞',\
                '∙', '）', '↓', '、', '│', '（', '»','，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø','¹', '≤', '‡', '√', '«',\
                '»', '´', 'º', '¾', '¡', '§', '£', '₤']:
        item = item.replace(pun, ' ')
    return item

# Defining Funciton removing non ascii characters 
def removing_non_ascii(item):
    item =str(item)
    item = item.encode('ascii', 'ignore')
    item = item.decode()
    return item

# Defining Funciton removing stopwords 
def stop_words():
  stop = stopwords.words('english')
  return stop

# Defining Function to Lemmatize data
def lemmatize(): 
  w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
  lemmatizer = nltk.stem.WordNetLemmatizer()
  return w_tokenizer,lemmatizer


# joining the list to string.
def string_join(item):
    item =' '.join(item)
    return item

def dropna_df(data_all):
  for data in data_all:
    data_all[data] =  data_all[data].dropna() 
  return data_all

def length_plot(data_all):
  length_news = []
  length_title = []
  for keys in data_all:
      for x in list(data_all[keys]['News']):
          length_news.append(len(x.split(' ')))

      for x in list(data_all[keys]['Title']):
          length_title.append(len(x.split(' ')))
  plt.hist(length_news,bins=100)
  plt.ylabel('Number of News Articles')
  plt.xlabel('length in words')
  plt.show()
  plt.hist(length_title,bins=10)
  plt.ylabel('Number of News Title')
  plt.xlabel('length in words')
  plt.show()
def min_count(data_all):
  min = 1000000000
  for keys in data_all:
    if data_all[keys].shape[0]<min:
      min = data_all[keys].shape[0]
  return min