
# Muthoni Portfolio

## Project 1: [Outlier Detection Function](https://github.com/Muthoni-w/mw_projects/blob/main/outlier_fxn.py)
Outlier detection is used to detect data that is anomalous and does not fit the normal statistical distribution of a dataset. Outliers can skew the data and therefore impact model effectiveness. They can be dealt in a variety of ways such as deletion, transformation e.t.c. Using the pandas library, I created an outlier detection function using IQR based filtering to identify these outliers for continuous numerical variables.
- iqr = q75 - q25
- lower_limit = q25 - iqr_multiplier * iqr
- upper_limit = q75 + iqr_multiplier * iqr

The results indicate that there are 116 outliers in the titanic dataset based on the Fare column. 


## Project 2: [Sentiment Analysis](https://github.com/Muthoni-w/mw_projects/blob/main/sentimant_analysis.py)
This is the use of NLP to identify emotions or attitudes towards a topic. Where expressions can be classified as ‘positive’, ‘negative’ or ‘neutral’ and give companies insight on how to enhance the customer experience. I used Open AI’s GPT 3 pretrained model to identify the sentiment of Amazon phone reviews.
GPT-3 is a Generative Pre-trained language Model that uses deep learning to predict the next token. For sentiment analysis, I have provided an instruction text to show it what I expect it to do. Since this is a simple task, the model is easily able to understand what is expected of it.

![](/images/label_3.PNG)

These are the parameters used. I used the Davinci model, although the curie model would still be able to perform this task. The Temperature setting has been set to 0 since the random aspect is not necessary for this specific dataset.

![](/images/label_1.PNG)

The results indicate that a majority of the customers gave a positive review on the Xiaomi phones for the year 2018. With 34 giving a positive review, 5 neutral and 7 negative.
![](/images/label_2.png)

## Project 3: [Feature Selection using Random Forests](https://github.com/Muthoni-w/mw_projects/blob/main/classification.py)
Feature selection is important in selecting relevant independent variables for a machine learning model. I used Random Forests to rank the importance of variables from the Titanic dataset. Feature selection is important for machine learning models to improve the learning accuracy by eliminating redundant and irrelevant features.
Here's the order of feature importance.
![](/images/label_4.png)
