# engr-ALDA-fall2021-P09

## Data Triage For News Articles Classification 

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
2. Clone the Git Repo
3. Keep all the downloaded folder into the git folder.
4. to run the whole process.  
    -   Use the pipline.py file This will perform Classification on the whole dataset folloed by Similarity Detection and followed by performing the classification again with Triage Data.
6. To the files individually.
    -    model.py can be used to run the Classification models 
    -    The different Similarity files can be use to run the similarity detection individually.
7. Data Cleaning script are present can be used using the code in data_cleaning_process.py file.
