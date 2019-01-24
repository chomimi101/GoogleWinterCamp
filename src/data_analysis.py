#import common library
import numpy as np
import pandas as pd

import numpy as np
import collections
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from data_clean import *
from ml_words_model import  *


def addNumericalFeatures(df):
    df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
    df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)
    df['music_per_comment'] = df['posts'].apply(lambda x: x.count('music')/50)
    df['question_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)
    df['img_per_comment'] = df['posts'].apply(lambda x: x.count('jpg')/50)
    df['excl_per_comment'] = df['posts'].apply(lambda x: x.count('!')/50)
    df['ellipsis_per_comment'] = df['posts'].apply(lambda x: x.count('...')/50)
    map1 = {"I": 0, "E": 1}
    map2 = {"N": 0, "S": 1}
    map3 = {"T": 0, "F": 1}
    map4 = {"J": 0, "P": 1}
    df['I-E'] = df['type'].astype(str).str[0]
    df['I-E'] = df['I-E'].map(map1)
    df['N-S'] = df['type'].astype(str).str[1]
    df['N-S'] = df['N-S'].map(map2)
    df['T-F'] = df['type'].astype(str).str[2]
    df['T-F'] = df['T-F'].map(map3)
    df['J-P'] = df['type'].astype(str).str[3]
    df['J-P'] = df['J-P'].map(map4)
    return df


def extractTotalPosts(df):
    df['posts_all'] = df['posts'].apply(lambda x: ' '.join(x.split('|||'))).apply(lambda x: cleanPost(x))
    return df

def extractPosts(df):
    df['posts'] = df['posts'].apply(lambda x: x.split('|||'))
    df['posts'] = df['posts'].apply(lambda x: [cleanPost(post) for post in x])
    df['words'] = df['posts'].apply(lambda x: [post.split() for post in x])
    df['words_count'] = df['posts'].apply(lambda x: np.sum([len(post.split()) for post in x]))
    return df

def convertSingleLabel(labels):
    unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                        'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    lab_encoder = LabelEncoder().fit(unique_type_list)
    labels = lab_encoder.transform(labels)
    return labels

def extractPostsAndLabels(df):
    posts, lables = [], []
    posts = df.posts.values
    lables = convertSingleLabel(df.type.values)
    return posts, lables

def runMLModel(df):
    df['posts'] = df['posts'].apply(lambda x: x.replace('|||', ' '))
    df['posts'] = df['posts'].apply(lambda x: cleanPost(x))
    posts, lables = extractPostsAndLabels(df)
    model = MLModel(posts, lables)
    model.set_nb_model()
    model.trian_model()
    model.set_lr_model()
    model.trian_model()

def drawWordsCountDistribution(df):
    df['posts'] = df['posts'].apply(lambda x: x.replace('|||', ' '))
    df['posts'] = df['posts'].apply(lambda x: cleanPost(x))
    posts = df.posts.values
    import seaborn as sns
    df["wordsCount"] = df["posts"].apply(lambda x: len([word for word in x.split() if word]))
    sns.distplot(df["wordsCount"]).set_title("Distribution of Words Count of all 50 Posts")



def test():
    df = pd.read_csv('data/mbti_1.csv')
    #df = addNumericalFeatures(df)
    # df = extractPosts(df)
    # print(df['words_count'].describe())
    #df = extractTotalPosts(df)
    runMLModel(df)
    #print(df['posts_all'].sample(1).values)



if __name__ == '__main__':
    test()







