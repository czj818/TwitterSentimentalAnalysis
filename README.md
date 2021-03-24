# TwitterSentimentalAnalysis

##Detect Offensive Tweets in Multilingual Context Using TFIDF and Ensemble Stacking Model

Nowadays with the prevalence of smartphone everyone is able to share ideas and comments online. This new era also comes with the rise of social media platforms such as Twitter. As of October 2019, every second, on average, around 6,000 tweets are sent, which corresponds to 500 million tweets per day. This makes it impossible to detect tweets with offensive language with human works. This is the time for us to bring machine learning into the game.

In this article, we will show how we develop a machine learning algorithm that can be used to detect offensive language in multilingual context.

## Introduction

Today, the high accessibility to social media anonymity feature in commenting and more freedom of speech all contribute to more offensive comments online. Commenting behind the computer gives people more courage to say aggressive and offensive languages without being responsible for that.

Cyberbully becomes a main issue on social media towards users. Because of the high accessibility to social media, those offensive tweets will have negative effect on the young generation. The detection of offensive tweets will build a better internet environment and protect social media users from being cyber bullied.

This task is a binary classification problem and in experiment, I will walk you through the steps I take to construct and evaluate models that can help achieve the goal. In the first part, I will talk about the method I choose to pre-process my data. Then I will show you multiple models I use including stacking ensemble model. Finally, we will discuss the performance of each model and whether we are able to find the best model from them.

## Data

The data we use is from an online competition called OffensEval 2020. Thanks to the organizer of this competition, we have a multilingual dataset with five languages:
- Arabic
- Danish
- English
- Greek
- Turkish

## Step 1 Data Inspection
Each dataset contains the content of each tweet and also the tag indicating whether it is offensive or not. After reading the data, I find a potential risk. Our data is imbalanced. The major risk of this issue is that we might fall into the trap of accuracy. For instance, let’s suppose in a dataset we have 100 objects with 90 positive and 10 negative. Even we make a prediction that all objects are positive, we still have an accuracy of 0.9. However, our prediction is bad because we fail to capture any negative object.

Here we create a function to help us read the file. There are in total more than 10 millions tweets in English dataset. I take a sample of 50,000 from English dataset and this makes our English dataset as big as other datasets.

![Alt Text](https://github.com/czj818/TwitterSentimentalAnalysis/blob/main/EDA_boxplot.jpeg)

## Step 2 Handle Imbalanced Data

In general, let’s suppose we have 1,000 offensive tweets and 2,000 non-offensive tweets. By using down-sample method we randomly choose 1,000 non-offensive tweets without replacement so that we now have a total of 2,000 tweets. On the other hand, using up-sample method means we randomly choose 2,000 offensive tweets with replacement and now we will have a dataset of 4,000 tweets. Either way we have a balanced data. Which way is better? The answers varies all the time so that we will find it out in following parts.

## Step 3 Preprocessing Data

Lucky for us, in NLTK package we have stop words list for all language we are analyzing here.
TfidfVectorizer is a powerful tool to use in text analysis. Traditional CountVectorizer implements both tokenization and count of occurrence. However, in a corpus, several common words makes up lot of space but carry very little information about content of document. If we feed these count straightly to our model, those common words can affect our model detecting real insightful information of the document. Using tf-idf transform method can enable us re-weight count feature vectors and we now have a better vectorized data to feed our classification model.
In our code, we create three different functions to apply TfidfVectorizer. We first split our dataset into training set and testing set. Then we do three things separately on our training set. For the first training set we choose to keep it constant. Then we apply down-sample method on the second training set and up-sample on the third training set so that in the following sections we are able to discuss the effectiveness of these methods.

## Model

## Step 4 Model Comparison

First, we will try 5 different models as our base models and also try to see what we can do next in order to improve the result.
The 5 models we choose are:
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- XGBoost
- Linear SVC
We first run our Arabic dataset without any up-sampling or down-sampling method to see both the accuracy and F1 score of our model.


