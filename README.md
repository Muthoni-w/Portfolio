
# Muthoni Portfolio

## Project 1: [Outlier Detection Function](https://github.com/Muthoni-w/Portfolio/blob/main/outlier_fxn.py)
Outlier detection is used to detect data that is anomalous and does not fit the normal statistical distribution of a dataset. Outliers can skew the data and therefore impact model effectiveness. They can be dealt in a variety of ways such as deletion, transformation e.t.c. Using the pandas library, I created an outlier detection function using IQR based filtering to identify these outliers for continuous numerical variables.
- iqr = q75 - q25
- lower_limit = q25 - iqr_multiplier * iqr
- upper_limit = q75 + iqr_multiplier * iqr

The results indicate that there are 116 outliers in the titanic dataset based on the Fare column. 


## Project 2: [Sentiment Analysis](https://github.com/Muthoni-w/Portfolio/blob/main/sentimant_analysis.py)
This is the use of NLP to identify emotions or attitudes towards a topic. Where expressions can be classified as ‘positive’, ‘negative’ or ‘neutral’ and give companies insight on how to enhance the customer experience. I used Open AI’s GPT 3 pretrained model to identify the sentiment of Amazon phone reviews.
GPT-3 is a Generative Pre-trained language Model that uses deep learning to predict the next token. For sentiment analysis, I have provided an instruction text to show it what I expect it to do. Since this is a simple task, the model is easily able to understand what is expected of it.

![](/images/label_3.PNG)

These are the parameters used. I used the Davinci model, although the curie model would still be able to perform this task. The Temperature setting has been set to 0 since the random aspect is not necessary for this specific dataset.

![](/images/label_1.PNG)

The results indicate that a majority of the customers gave a positive review on the Xiaomi phones for the year 2018. With 34 giving a positive review, 5 neutral and 7 negative.
![](/images/label_2.png)

## Project 3: [Feature Selection using Random Forests](https://github.com/Muthoni-w/mw_Portfolio/blob/main/classification.py)
Feature selection is important in selecting relevant independent variables for a machine learning model. I used Random Forests to rank the importance of variables from the Titanic dataset. Feature selection is important for machine learning models to improve the learning accuracy by eliminating redundant and irrelevant features.
Here's the order of feature importance.
![](/images/label_4.png)

## Project 4: [Revenue Forecast using AutoArima](https://github.com/Muthoni-w/Portfolio/blob/main/forecast.ipynby)
ARIMA(Autoregressive integrated moving average) is a statistical analysis model that uses time series data to predict future values based on the fact that past values have sufficient information.
Parameters in the model are p, q and d.
p - Order of the AR term
q - Order of the MA term
d - Order of differencing

AR is a linear regression model that uses its own lags as predictors. This can be obtained using the PACF between the series and its lag.

MA refers to the no. of lagged forecast errors that should go into a model. This can be obtained using the ACF that tells us how many moving average terms are required to remove any autocorrelation in the series.

d refers to the No. of differencing required to make the time series stationary. This can be obtained using the ADF test. With the null hypothesis of the ADF test being that the time series is non-stationary. So if the null hypothesis is less than the significance level (0.05) then you reject the null hypothesis and conclude that the time series is indeed stationary.

Once p, d and q have been obtained, and ARIMA model can be built. The significant coefficients can be identified when the p-value is less than 0.05, and the AIC and BIC are at their lowest value.

Auto-arima on the other hand applies automated configuration tasks to the ARIMA model to search multiple combinations of p,d,q parameters and chooses the best model that has the least AIC.

Based on the results obtained from the revenue forecast on the Bikes data during the period from 2010-2015, it is clear that the Best model is that with a p, d and q of 3,0 and 0 respectively.        
 
![](/images/Capture_5.PNG)

Here's the revenue forecast visualization.
![](/images/output_6.png)

