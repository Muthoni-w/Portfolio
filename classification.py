# Feature Selection using Random Forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('titanic_train.csv')

train_df.head()

train_df.isna().sum()

mode = train_df.mode().iloc[0]

def data(train_df):
    train_df['Fare'] = train_df.Fare.fillna(0)
    train_df.fillna(mode, inplace=True)
    train_df['LogFare'] = np.log1p(train_df['Fare'])
    train_df['Embarked'] = pd.Categorical(train_df.Embarked)
    train_df['Sex'] = pd.Categorical(train_df.Sex)

data(train_df)

#splitting into continous and categorical variables
cats=["Sex","Embarked"]
conts=['Age', 'SibSp', 'Parch', 'LogFare',"Pclass"]
dep="Survived"

#splitting into test and train dataset
random.seed(42)
trn_df,val_df = train_test_split(train_df, test_size=0.25)

#Encode categorical variables
trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)

trn_df.shape
val_df.shape

#Function to split independent and dependent variable
def xs_y(df):
    xs = df[cats+conts].copy()
    return xs,df[dep] if dep in df else None

trn_xs,trn_y = xs_y(trn_df)
val_xs,val_y = xs_y(val_df)

#Measure accuracy
rf = RandomForestClassifier(100, min_samples_leaf=5)
rf.fit(trn_xs, trn_y)
mean_absolute_error(val_y, rf.predict(val_xs))

#Feature Selection using Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(trn_xs, trn_y)

feature_scores = pd.Series(clf.feature_importances_, index=trn_xs.columns).sort_values(ascending=False)
feature_scores

# Creating a seaborn bar plot
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.show()