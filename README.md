# Twitter_Glove
The positive/negative classification of the twitter feedbacks for products.
# Requirements:
* Numpy
* Pandas
* nltk(Natural Language Toolkit)
* re(for Regular Expression Operations)
* Glove
* TSNE(For mapping the features in high-dimensions to 2-D )
* SMOTE(For synthetic generation of feature vectors)
# About:
Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations. Brands can use this data to measure the success of their products in an objective manner. Here I tried to predict the sentiments of the customers on electronic products.
* First I tried to clean and pre-process the feedback lines in the dataset and tokenized the words.
* Then I tried to use SnowBall Stemmer for stemming the sentences, one may use Lemmatizer to try different approach.
* Then I tried to remove the Stopwords which appear very frequently and may not be of much use.
* After this I extracted the feature vectors for the sentences of 100 dimensions by using a pre-trained Glove Model.
* As our dataset was imbalanced I tried SMOTE for generation of feature vectors for the minority class to balance out the support for each class , one may just replicate the minority class but it may cause the classifier to have a bias towards that type of example.
* After this , I used the XGBoost classifier for getting the classification score.
* Next I tried to use LSTM architecture on the same upsampled dataset and it can be seen that the loss decreases with the epoch and the validation accuracy is more than the training accuracy while the validation loss being lesser that the training loss which may be due to the regularizing effect of the Dropout used.
# How to Use:
Clone the current repository and load any twitter sentiment(binary) dataset for classification of the sentiments by making appropriate changes.
# Result:
Here I tried to evaluate the ML models (Glove+SVC, Glove+ XGBoost) and deep learning (LSTM+Dense Layer) architecture over this small imbalanced dataset , it can be seen in the classification score in nlp_glove_smote.ipynb that the Glove+XGBoost Classifier gave a weighted F1 score of 0.92 while the LSTM architecture gives F1 score of 0.86 on the validation dataset, I tried different epochs and batch while training but the ML models gave a better classification score.
