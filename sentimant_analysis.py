#Import relevant libraries
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

# Read amazon data
items = pd.read_csv('items.csv')
reviews = pd.read_csv('reviews.csv')

items.head()
reviews.head()

#join columns
phone_data = reviews.merge(
    right = items,
    how = 'left',
    left_on = 'asin',
    right_on = 'asin'
)
phone_data.head()
phone_data.columns

# Extract year
phone_data['date'] = pd.to_datetime(phone_data['date'])
phone_data['year'] = pd. DatetimeIndex(phone_data['date']). year

cols_to_keep = [
    "asin",
    'name',
    'verified',
    'date', 
    'price',
    'title_y',
    'year',
    'title_x',
    'brand',
    'price'
    ]

df = phone_data[cols_to_keep]

#drop unecessary columns
df = df.dropna(subset=['brand'])

data = df[["title_x", "year", "brand"]]
brand_name = data[data["brand"] == "Xiaomi"]
final = brand_name[brand_name["year"] == 2018]

input_text = final['title_x']

#GPT-3 Sentiment Analysis
input_data=input_text.to_csv('input_text.csv', index = None, header=0)
input_data = "input_text.csv" #Select appropriate path to file
output_data= "output_text.csv" #Select appropriate path to file

#prompt function
def generate_prompt(sentiment):
    return """Decide whether an amazon review's sentiment is positive, neutral, or negative.

###
Review: {}
Sentiment:""".format(
    sentiment
    )

header = "review"+","+"sentiment"
with open(output_data, 'w') as n, open(input_data, 'r') as o:
    writer = csv.writer(n, quoting=csv.QUOTE_ALL)
    reader = csv.reader(o)
    writer.writerow(header.split(","))
    for row in reader:
        input = str(row)[2:-2].strip()
        prompt = generate_prompt(input)
        print(prompt)
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["###"]   
        )
        output = response["choices"][0]["text"].strip().split("\n")
        writer.writerow(row + output)

#3.0 Visualize Data
df2 = df.groupby(['sentiment'])['sentiment'].count().reset_index(name="count")
df = pd.read_csv('output_text.csv')

fig = plt.figure(figsize = (10, 5))
plt.bar(df2["sentiment"], df2["count"], color ='maroon',
        width = 0.4)
 
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Count")
plt.show()