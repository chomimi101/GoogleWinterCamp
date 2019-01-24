import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


METRIC = { 'acc': 'accuracy',
           'neg_log_loss': 'neg_log_loss',
           'f1_micro': 'f1_micro'}

class MLModel:

    def __init__(self, X, y):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        self.tfidf2 = CountVectorizer(ngram_range=(1, 2),
                                      stop_words='english',
                                      lowercase=True,
                                      max_features=5000)
        self.tsvd = TruncatedSVD(n_components=10)
        self.kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        self.model = None
        self.X = X
        self.y = y


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    def trian_model(self):
        if self.model is None:
            return
        print("start train")
        results = cross_validate(self.model, self.X, self.y, cv= self.kfolds, scoring= METRIC, n_jobs=-1)
        self.print_model_results(results)

    def set_nb_model(self):
        np.random.seed(1)
        self.model = Pipeline([('tfidf1', self.tfidf2), ('nb', MultinomialNB())])


    def set_lr_model(self):
        np.random.seed(1)
        self.model = Pipeline([('tfidf1', self.tfidf2), ('lr', LogisticRegression(class_weight="balanced", C=0.005))])

    def print_model_results(self, results):
        print("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results['test_acc']),
                                                          np.std(results['test_acc'])))

        print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results['test_f1_micro']),
                                                    np.std(results['test_f1_micro'])))

        print("CV Logloss: {:0.4f} (+/- {:0.4f})".format(np.mean(-1 * results['test_neg_log_loss']),
                                                         np.std(-1 * results['test_neg_log_loss'])))


    def train_models(self, name):

        if name.startwith('sgd'):
            model = SGDClassifier(n_iter=5)
        elif name.startwith('rf'):
            model = RandomForestClassifier(n_estimators=100)
        elif name.startwith('lr'):
            model = RandomForestClassifier(n_estimators=100)
        elif name.startwith('knn'):
            model = KNeighborsClassifier(n_neighbors=5)


        model.fit(self.X_train, self.y_train)
        Y_pred = model.predict(self.X_test)
        model.score(self.X_train, self.y_train)
        print("run model ", name)
        acc = round(model.score(self.X_train, self.y_train) * 100, 2)
        print("Training Data Set", round(acc, 2, ), "%")
        acc = round(model.score(self.X_test, self.y_test) * 100, 2)
        print("Testing Data Set", round(acc, 2, ), "%")



def bagOfWordsModel(df):
    vect = CountVectorizer(stop_words='english')
    X = vect.fit_transform(df.posts)

    le = LabelEncoder()
    y = le.fit_transform(df.type)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)