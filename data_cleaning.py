import DataCleaning as dc
import pandas as pd

def cleaning_data(data,news_title):

  data[news_title] = data[news_title].str.lower().apply(dc.puncuation)
  data = dc.dropna_df(data)

  stop = dc.stop_words()
      
  data[news_title] = data[news_title].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
  data = dc.dropna_df(data)

  data[news_title] = data[news_title].str.replace('\d+', '')
  data = dc.dropna_df(data)

  data[news_title] = data[news_title].apply(dc.removing_non_ascii)
  data = dc.dropna_df(data)

  w_tokenizer,lemmatizer = dc.lemmatize()
  def lemmatize_text(text):
      return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

  data[news_title] = data[news_title].apply(lemmatize_text)
  data = dc.dropna_df(data)

  data[news_title] = data[news_title].apply(dc.string_join)
  if news_title == 'News':
    data[news_title] = data[data[news_title].str.count(' ') >= 64][news_title]
  elif news_title == 'Title':
    data[news_title] = data[data[news_title].str.count(' ') >= 4][news_title]
  data = dc.dropna_df(data)

  if news_title == 'News':
    data[news_title] = data[news_title].apply(dc.length_of_news)
  elif news_title == 'Title':
    data[news_title] = data[news_title].apply(dc.length_of_title)
  data = dc.dropna_df(data)

  return data