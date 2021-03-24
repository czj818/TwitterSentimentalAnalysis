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


