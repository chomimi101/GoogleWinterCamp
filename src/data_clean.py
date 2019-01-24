from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

REMOVE = ''
replace_special = {
   r'[^a-zA-Z\s]': REMOVE, #Remove Non-words
   r'([a-z])\1{2,}[\s|\w]*':REMOVE, #Remove multiple letter repating words
   r'(\b\w{30,1000})?\b':REMOVE #Remove short/long words
}
#alwayyyyyyys
#emoj
def cleanWebText(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    return text

def textNormalization(text):
    stemmer = SnowballStemmer("english")
    text = " ".join(stemmer.stem(p) for p in text.split())

def cleanPost(post):
    post = post.lower()
    post = post.replace('|||', ' ')
    for ori, rep in replace_special.items():
        post = post.replace(ori , rep)
    post = cleanWebText(post)
    return post


#test = df.sample(10)
#extractPosts(test)

