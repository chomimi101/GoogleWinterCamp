# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
PW = 8
PH = 6
plt.rcParams['figure.figsize'] = (PW, PH)
plt.rcParams['image.cmap'] = 'gray'

import re
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_rows = 250
pd.options.display.max_columns = 500
pd.options.display.max_colwidth = 500

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import string
import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#
# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv("./data/mbti_1.csv", encoding="utf-8")
print("Number of users", len(df))

#Personality Types
groups = df.groupby("type").count()
groups.sort_values("posts", ascending=False, inplace=True)
print ("Personality types", groups.index.values)

#Priors used below for Random Guessing Estimation
priors = groups["posts"] / groups["posts"].sum()


def preprocess_text(df, remove_special=True):
    # Remove links
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|", " ") + " "))

    # Keep EOS
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))

    # Strip Punctation
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Remove Non-words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # To lower
    df["posts"] = df["posts"].apply(lambda x: x.lower())

    # Remove multiple letter repating words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*', '', x))

    # Remove short/long words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{0,3})?\b', '', x))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{30,1000})?\b', '', x))

    # Remove Personality Types Words
    # This is crutial in order to get valid model accuracy estimation for unseen data.
    if remove_special:
        pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ', 'ISFJ',
                      'ESTP', 'ESFP', 'ESFJ', 'ESTJ']
        pers_types = [p.lower() for p in pers_types]
        p = re.compile("(" + "|".join(pers_types) + ")")

    df["posts"] = df["posts"].apply(lambda x: p.sub(' PTypeToken ', x))
    return df


# Used for class balancing. When class balancing is used dataset becomes very small.
def subsample(df):
    groups = df.groupby("type").count()
    groups.sort_values("posts", ascending=False, inplace=True)

    min_num = groups["posts"][-1]
    min_ind = groups.index[-1]
    ndf = df[df["type"] == min_ind]

    for pt in groups.index[:-1]:
        print(min_num, pt)
        tdf = df[df["type"] == pt].sample(min_num)
        ndf = pd.concat([ndf, tdf])
    return ndf

#Number of Posts per User
df["NumPosts"] = df["posts"].apply(lambda x: len(x.split("|||")))

#Split to posts
def extract(posts, new_posts):
    for post in posts[1].split("|||"):
        new_posts.append((posts[0], post))

posts = []
df.apply(lambda x: extract(x, posts), axis=1)
print("Number of users", len(df))
print("Number of posts", len(posts))

df = pd.DataFrame(posts, columns=["type", "posts"])

#Preprocess Text
df = preprocess_text(df)

#Remove posts with less than X words
min_words = 15
print("Number of posts", len(df))
df["nw"] = df["posts"].apply(lambda x: len(re.findall(r'\w+', x)))
df = df[df["nw"] >= min_words]
print("Number of posts", len(df))

df["Len"] = df["posts"].apply(len)

#Remove long post
max_length = 500
print("Number of posts", len(df))
df = df[df["Len"] < 500]
print("Number of posts", len(df))

#Stem
stemmer = SnowballStemmer("english")

df["posts"] = df["posts"].apply(lambda x: " ".join(stemmer.stem(p) for p in x.split(" ")))

for i in range(df.shape[0]):
    info = df.iloc[i]


print(df.iloc[np.random.choice(len(df),10),:])