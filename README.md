# TwitterSentimentalAnalysis

##Detect Offensive Tweets in Multilingual Context Using TFIDF and Ensemble Stacking Model

Nowadays with the prevalence of smartphone everyone is able to share ideas and comments online. This new era also comes with the rise of social media platforms such as Twitter. As of October 2019, every second, on average, around 6,000 tweets are sent, which corresponds to 500 million tweets per day. This makes it impossible to detect tweets with offensive language with human works. This is the time for us to bring machine learning into the game.

![Alt Text](https://www.niemanlab.org/images/wordle_wordle-300x160.png)

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

![Alt Text](https://github.com/czj818/TwitterSentimentalAnalysis/blob/main/img/arabic_acc.png)

We can see that from the box plot above, all models seem to achieve a very good accuracy. Our logistic regression model has the highest accuracy, which equals to 0.839. Even the worst model, which is XGboost, also achieves an accuracy of 0.762. However, does this mean we already in a good place now? Since we have an imbalanced data, we also need to check F1 score for these models.

![Alt Text](https://github.com/czj818/TwitterSentimentalAnalysis/blob/main/img/arabic_f1.png)

This time we see that logistics regression model turns into our worst model in terms of F1 score. It only has a F1 score of 0.588. Linear SVC model has the best F1 score, which equals to 0.712, which is ok but we hope we can better improve this number.
From the analysis in this part, we have several takeaways:
1. Imbalanced data does cause a trap in accuracy. We can no longer use accuracy as our main metric to compare models.
2. Different models show different performances so that we might need to choose different models for different languages.

## Step 5 Ensemble Stacking Model

The idea behind ensemble stacking model is relatively easy to understand. Graph below shows a two-level ensemble stacking model. In the first layer we have three different models each generating predictions after fitting on the training set. Instead of directly using the prediction from these model, we introduce a second layer model which take the predictions from first layer as input and produce our final result.

![Alt Text](https://github.com/czj818/TwitterSentimentalAnalysis/blob/main/img/stacking.png)

Stacking is very popular and has won many predictive model competition. Depending on the scenario, some data scientist can even train stacking models with more than 2 layers. Generally speaking, stacking can always bring improvements to the predictive model, sometimes a huge one.
The reason is that stacking can combine many weak learners together and become a strong learner. One good analogy I can think of is that if the professor asks a yes or no question to an individual student during the lecture, he or she might answer it incorrectly. However, if the professor asks every student writes answer on the paper and check the final votes, normally the class will more likely to answer it correctly because this time the answer is a combination of every student’s wisdom.

## Results and Discussions

## Step 6 Evaluate the results

Since we discussed in the previous section that accuracy can be misleading, we will use F1 score as our main metric to compare models. 

Both up-sampling/down-sampling and stacking bring improvement to our dataset. In all 5-language analysis, we see that a stacking model with up-sampling or down-sampling method yields the best result. If we take a look at our Greek dataset, we can see that stacking method has a big improvement in terms of F1 score compared to only running a single model.
Also, another benefit of stacking method is that we do not need to worry about which model we should use. We can see in some languages Linear SVC model is our best choice if we only consider a single model. However, in some other languages, random forest also has a strong performance. But if we use stacking method, because it always generate a great result as a combination of many models, we are guaranteed to have a good result.

## Step 7 Conclusion

We see that 4 out of 5 times, our up-sampling stacking model gives us the best result in terms of F1 score and both stacking method and up-sampling/down-sampling method always help.
Based on what is shown from the model, we have several key takeaways:
1. When we encounter an imbalanced dataset, we should not use accuracy to evaluate the data.
2. Up-sampling/down-sampling is a useful tool to fix the potential risk of imbalanced data. However, we should always run both methods to decide which one we want to apply.
3. Ensemble stacking method have many advantages. First, it can improve the final result because a strong learner is built from several weak learners. Second, we do not need to worry about which model we should use. Stacking method can always give us a great model.

