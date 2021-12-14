
# Data Triage In News Articles Classification 

### Background
Data is one of a company’s most valuable resources. The extraction of meaningful information fromdata to enhance business analytics has grown increasingly time intensive as data collection activities have increased across all sectors. Data triage is a technique for reducing the size of a dataset in order to achieve equivalent results with less computational time and resources. The goal of this project is to perform Data Triage on News Articles in order to minimize the original dataset while maintaining similar level of accuracy.

### Proposed Method
#### Approach
Our approach for data triage:
- The original data is divided into training and testing sets. We initially train our models using the unaltered training dataset and perform classification of the test dataset.
- Document similarity algorithms are implemented on the training data set to produce our training subsets. For each similarity algorithm, we set different thresholds based on our knowledge of the model and a series of experiments of the model.

&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/22122136/145949654-a765a9da-a885-4163-9e1d-637acc3271ba.png" width="250" height="110">

- Post dataset reduction, we follow a similar pipelines of operations by training our model on the data subset and performing classification of the test data.

The following methods have been used for data classification:
- Logisitic Regression: It is a process of modeling the probability of a discrete outcome given an input variable. Logistic regression is a useful analysis method for classification problems, where you are trying to determine if a new sample fits best into a category. Logistic Regression was used with both TF-IDF vectorization method. The text is tokenized and converted into its corresponding vector before being fed into the logistic regression model which later predicts the class of the test news articles.
- Naive Bayes: A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem and the information of the probability of the word occurrences is given by our vectorization technique. The text is tokenized and converted into its corresponding vector before being fed into the bayesian classifier model.
- SVM (Support Vector Machine): The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points. Our objective is to find a plane that will help classify our news articles into different categories. SVM was used with TF-IDF vectorization which transformed the data into a vector format which could be used for classification.
- Long Short-Term Memory: A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this sense, neural networks refer to systems of neurons, either organic or artificial in nature. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria. The neural networks model is used for tokenization, text vectorization and as a classification model which runs for ten iterations learning from every iteration giving an optimal accuracy. 

The following methods have been used for text vectorization:
- TF-IDF: Term Frequency-Inverse Document Frequency (TF-IDF) is a common method to evaluate how important a single word is to a corpus. The TF-IDF method can assess the importance of each word in the corpus by taking into account the number of its occurrences. It can be obtained by multiplying the number of times a word appears in a document, and the inverse document frequency of the word across a set of documents. TF-IDF will output the vectors that are required for classification and similarity measures that are used in the project. 

The following methods have been used for detecting document similarity:
- Jaccard Similarity: Jaccard Similarity coefficient, also sometimes referred to as Jaccard Index, is Intersection over Union. For the project, we have used the Trigram approach where the entire article is divided into sets of 3 words, known as trigrams, and then the trigrams of the articles are compared for obtaining similarity.
- Cosine Similarity: Cosine Similarity is a measurement that quantifies the similarity between two or more vectors. The cosine similarity is the cosine of the angle between vectors [9]. Suppose the angle between the two vectors is 90 degrees, the cosine similarity will have a value of 0; this means that the two vectors are perpendicular to each other which means they have no correlation between them [10]. In our implementation, we have used Tf-Idf in conjunction with cosine similarity to identify similarity between the news articles to form the basis for subset creation.
- GloVe Embeddings: GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space [11]. In our application, we have used Glove Embedding alongside cosine similarity to identify similarity between the news articles to form the basis for subset creation. 
- Doc2Vec Embeddings: Doc2Vec is a powerful NLP tool to represent documents as a vector. Doc2Vec model is based on the Word2Vec model. While Word2Vec computes a feature vector for every word in the corpus. Doc2Vec computes a feature vector for every document in the corpus. Doc2Vec is also used with cosine similarity to create our training data subsets.

#### 2.2 Rationale
We examine whether two articles are sufficiently similar using different Document Similarity techniques, and whether eliminating one of them would have minimal effect on the performance of the classification models used. We can decrease the original dataset by comparing all of the News Articles to one another and deleting similar or roughly duplicate News Articles, resulting in the classification models requiring substantially lesser computational resources and time and provide nearly same accuracy as implemented on the original dataset.

### Plan and Experiment

We have used open source news articles datasets and one proprietary dataset from aylien. We collated articles for the following categories: Finance, Entertainment, Politics, Technology and Travel. The source data was originally available as .json files. For the ease of processing, we pre-processed and stored the data in .csv format. 

#### Data Sourcing
The dataset format for News Articles from webhose.io [1], [2], [3], [4] is as below:
Features: organization (string), uuid (string), author (string), url (string), ord\_in\_thread (string), title (string), locations (string), highlightText (string), language (string), text (string), published (date), crawled (date), highlightTitle (string)
The dataset format for Political News Articles from POLUSA [6] is as below:
Features: id (integer), date\_publish (date), outlet (string), headline (string), lead (string), body (string), body (string), authors (string), domain (string), url (string), political\_leaning (string) 
The dataset format for Financial News Articles from Aylien [5] is as below:

Features: author (string), body (string), categories (string), character\_count (integer), clusters, entities, hashtags, keywords (string), language (string), links (string), media (string), paragraph\_count (integer), published\_at (date), sentences\_count (integer), sentiment (string), social\_shares\_count (integer), source (string), summary (string), title (string), words\_count (integer)

#### Data Description

Features: Class (indicating the news category), Title (news article title), News (news article)

After dropping all the records with missing values, we obtained the clean dataset. The below table shows the number of records in each category: 

<img src="https://user-images.githubusercontent.com/22122136/145950986-8616a0c1-61ad-46a7-9338-148f191a7f94.png" width="250" height="150">

As visible in the above table, the record counts for Politics News Articles is much higher than the remaining categories. Thus, data balancing across the categories was required. 

The below graph represents the length of the news articles in the dataset. As visible in the graph, there are a lot of news articles with length of more than 512 words.Thus, for better computation on such a huge dataset, we trimmed the news articles body to a maximum of 512 words. Doing this would not affect the results of the classification substantially as 512 words would be enough for any news articles to be classified into one of the six categories in our dataset.

#### Data Preprocessing

From the raw data, only the columns ‘Title’, ‘News’ and ‘Category’ were retained since only these columns are relevant for the Data Triage and Classification.
On the dataset, we performed Data Sanitization in which the Non-ASSCII characters, Punctuations and Numeric characters were removed. Then all the News Articles with Body length less than 64 words or Title length less than 4 words were removed as shorter articles did not add significant value to the classification. Then, by using the NLTK library, Lemmatization was performed and Stop Words were removed from the Title and Body of each article. 
As the main goal of our project is to get good classification results on the Triaged dataset, we trimmed each article body to maximum of 512 words and each title to maximum of 20 words. To obtain the pre-processed data set on which we can perform classification, we balanced the data such that there were equal number of datapoints for each of the six categories of News we have. This is done so that the model can be trained without any bias towards a particular category which would have resulted from an unbalanced dataset.

<img src="https://user-images.githubusercontent.com/22122136/145951378-1e4610ce-c5f2-43d8-8b47-57817820aa60.png" width="450" height="150">

#### Data Summarization and Visualization

The below figure is a small representation of the preprocessed dataset. The text obtained after preprocessing is clean of all the Non-ASCII characters, punctuations, numeric characters and stop words.

<img src="https://user-images.githubusercontent.com/22122136/145951193-8c26f92c-e880-4dca-b734-ce55bb970997.png" width="250" height="150">

#### Hypothesis

The hypothesis for our experiment is to obtain similar results with a smaller training dataset which forms the basis for Data Triaging. It states that a smaller training dataset may provide equivalent if not more information to the models to help them correctly classify the News Articles into our given categories. Training the models on a subset of the data is expected to decrease the computational operations which may introduce a trade-off with a reduction in accuracy. The main focus of this project is to identify the best method for data reduction that will retain the most information provided by our training dataset. Additionally, we want to determine the decline in the accuracy if any of our classification models when training with a smaller dataset and decide if the trade-off between accuracy and computational time is warranted.

#### Experiment Setup

The below flowchart demonstrates the experimental setup used to perform Data Triage methods and Classification on the preprocessed dataset. First, the preprocessed data is split into training and testing data in the ratio 80:20. Classification algorithms described above are implemented and accuracy is obtianed using the testing dataset. Then, the Document Similarity models are executed on the same training data and a reduced dataset (triaged dataset) is obtained. The same classification models are implemented on the new training data. The accuracy of the classification models are obtained using the same testing dataset. The results of each classification model before and after Data Triage are compared and the results are analyzed.

<img src="https://user-images.githubusercontent.com/22122136/145951461-f651eb81-2954-4c8f-b1c0-66812469f68c.png" width="250" height="350">

#### Results

We implemented Term Frequency - Inverse Document Frequency (TF-IDF) word vectorization with Logistic Regression, Naive Bayes Classification, Support Vector Machine Classification and Recurrent Neural Networks using Long-Short Term Memory (LSTM).

<img src="https://user-images.githubusercontent.com/22122136/145951548-e6c5206a-2515-401b-9918-93c4d911ff5b.png" width="250" height="150">

GloVe Embedding was found to be most effective in performing data triage.

<img src="https://user-images.githubusercontent.com/22122136/145951603-8a92e7b8-492f-4aab-a7d0-480b59f76bf6.png" width="250" height="150">

All classification algorithms were implemented on our reduced dataset, obtained using Tf-Idf + Cosine Similarity Method.

<img src="https://user-images.githubusercontent.com/22122136/145952055-be5a6ed7-f94f-4cd6-99cc-25bec27ca2f7.png" width="250" height="150">

Consequently, we implemented the classification algorithms for the reduced dataset obtained using all the similarity methods and obtained interesting results.

<img src="https://user-images.githubusercontent.com/22122136/145952094-53d35e5a-1a90-41ea-96a5-bbcc9d4089be.png" width="250" height="150">

#### Critical Evaluation

At the start of the project, we assumed that the model which can best detect the contextual similarity between news articles would be the best technique for the Data Triage as it would remove the articles which would add no significance to the classification model's training process. So, we assumed that GloVe Embeddings would perform the best for our problem statement. However, in the results we obtained, we observed that Data Triage using GloVe Embeddings had the least testing accuracy for all classification models.
We address the decline in accuracy by considering the performance of our SVM model with the subset created by Tf-Idf and Consine Similarity Method. It has the highest accuracy and only drops seven percent compared to the accuracy obtained by our original dataset.
Given that the above similairty method causes triaging of about 35%, considerably reducing the computational operations required for training our our model, the accuracy drop seems justifiable.
However, this conclusion of a warranted trade-off is an open ended question and the answer to it depends on the use-case.

#### Conclusion 
- We obtained a better grasp of the working of Document Similarity Algorithms such as Jaccard Similarity, GloVe Embeddings, TF-IDF, and Doc2Vec Model during the term project on Data Triage for News Articles Classification. We were able to identify the algorithms' flaws and how other algorithms would address them.
- GloVe was efficient in finding the contextual meaning (rather than the semantic meaning) between News Articles which gave it the upper hand in detecting Similar News Articles. However, a lot of articles were removed which decreased the classification accuracy post data triage.
- TF-IDF detects the ranking of words in different documents and using it in conjunction with the Cosine Similarity was able to retain most information of the training dataset and provide highest accuracy when classifying the articles in our test dataset.
- Bi-directional LSTM with GloVe Embeddings is an efficient classification model provided that there is sufficient training data.
- Due to the time constraint and the lack of computational resources required to implement Doc2Vec Model, we could not successfully obtain the results on the entire dataset. The code for the same is available on our Github repository. However, we tested it on a small portion of the dataset and found it to be an effective model for detecting document similarity. As a result, we believe it can be a useful tool for performing data triage on large datasets if appropriate computational resources and time are available.

#### References
[1] Rajeshwari M R & Dr. Kavitha K S (2017) Reduction of Duplicate Bugs and Classification of Bugs using a Text Mining and Contingency Approach. International Journal of Applied Engineering Research

[2] Simon Rodier & Dave Carter (2020) Online Near-Duplicate Detection of News Articles Digital Technologies, National Research Council Canada

[3] Kowsari, Kamran, Kiana Jafari Meimandi, Mojtaba Heidarysafa, Sanjana Mendu, Laura Barnes, and Donald Brown. 2019. "Text Classification Algorithms: A Survey" Information 10, no. 4: 150. https://doi.org/10.3390/info10040150

[4] Susan Li (2019) Multi-Class Text Classification with LSTM, towardsdatascience.com

[5] Gunter Rohrich (2020) Find Text Similarities with your own Machine Learning Algorithm, towardsdatascience.com

[6] Jair Neto (2021) Best NLP Algorithms to get Document Similarity, medium.com

[7] Sindhu Seelam (2021) Machine Learning Fundamentals: Cosine Similarity and Cosine Distance, medium.com

[8] Jeffrey Pennington,   Richard Socher,   Christopher D. Manning (2014) GloVe: Global Vectors for Word Representation, nlp.stanford.edu

[9] Aylien News API Financial Crimes Dataset https://aylien.com/blog/free-dataset-downloads-natural-disasters-financial-crimes-nasdaq-100



### Prerequisite 

Create a Python environment with Python 3.6 
The libraries needed are
1. numpy 1.19.2
2. pandas 1.3.4
3. nltk 3.6.5
4. torch 1.11.0
5. keras 2.3.1
6. sklearn 1.0.1
7. statistics
8. gensim 4.1.2
9. tensorflow 2.0.0 

### How to run 

1. Download all the data files from google drive 
    -   https://drive.google.com/drive/folders/1sVaSB2Ala7Pfz47--eIVFp9NEcU7x-_p?usp=sharing  Data (Triage Data)
    -   https://drive.google.com/drive/folders/18EHAQoWQ8sDxp5vg2t5lWDt9TrN4w369?usp=sharing  data_class
    -   https://drive.google.com/drive/folders/1DA5Npe4nlksfIOPztgIkRru0BZ4zS98E?usp=sharing  data_class_clean
    -   https://drive.google.com/drive/folders/1vEZau-bsuNN0bzXKpUk0X9X26nSF0zQu?usp=sharing  data_class_limited_data
    -   https://drive.google.com/drive/folders/1gSAHndhHvNhAR1NOmkmtEnLy9BnBK5aX?usp=sharing  data_class_limited_data_train_test
    -   https://drive.google.com/drive/folders/1WsZCPGeMs7uasECOLlqJEWUCQ_hPOkI1?usp=sharing  Model files
2. Clone the Git Repository
3. Move all the downloaded data files into the git folder.
4. In order to run the whole process,  
    -   Execute the pipline.py file, which will perform Classification on the whole dataset followed by Similarity Detection and then repeating classification on the Triage Data.
    -   model.py can be used to run the Classification models 
    -   The different Similarity files can be use to run the similarity detection individually.
7. Data Cleaning can be performed using the data_cleaning_process.py file.
