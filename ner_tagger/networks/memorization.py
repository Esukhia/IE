# %% In [1]:
!pip install comet_ml
!pip install seqeval[gpu]

# %% In [2]:
# import comet_ml in the top of your file
from comet_ml import Experiment

# %% In [3]:
# Dataset
import pandas as pd
import numpy as np

# %% In [4]:
dataset_name = 'citation'

# %% In [5]:
data = pd.read_csv(f"../input/ner-lists-87_citations-1000_{dataset_name}.csv")
data = data.fillna(method="ffill")
data.tail(10)

# %% In [6]:
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None

# %% In [7]:
getter = SentenceGetter(data)

# %% In [8]:
tagged_sentences = getter.sentences
tagged_sentences[0]

# %% In [9]:
sentences, tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tag = list(zip(*tagged_sentence))
    sentences.append(list(sentence))
    tags.append(list(tag))

# %% In [10]:
from itertools import chain

# %% In [11]:
X = list(chain.from_iterable(sentences))
y = list(chain.from_iterable(tags))

# %% In [12]:
len(X), len(y)

# %% In [13]:
# Model Definition
model_name = 'Memorization'

# %% In [14]:
params = {
    'model_type': model_name
}

# %% In [15]:
from sklearn.base import BaseEstimator, TransformerMixin

# %% In [16]:
class MemoryTagger(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y):
        '''
        Expects a list of words as X and a list of tags as y.
        '''
        voc = {}
        self.tags = []
        for x, t in zip(X, y):
            if t not in self.tags:
                self.tags.append(t)
            if x in voc:
                if t in voc[x]:
                    voc[x][t] += 1
                else:
                    voc[x][t] = 1
            else:
                voc[x] = {t: 1}
        self.memory = {}
        for k, d in voc.items():
            self.memory[k] = max(d, key=d.get)
    
    def predict(self, X, y=None):
        '''
        Predict the the tag from memory. If word is unknown, predict 'O'.
        '''
        return [self.memory.get(x, 'O') for x in X]

# %% In [17]:
exp_name = f'{model_name}_v1'

# %% In [18]:
%%writefile .env
COMET_API_KEY=<HIDDEN>

# %% In [19]:
exp = Experiment(project_name="ner-citation-model")
exp.set_name(exp_name)

# %% In [20]:
exp.log_parameters(params)

# %% In [21]:
# Training
from sklearn.model_selection import cross_val_predict

# %% In [22]:
pred = cross_val_predict(estimator=MemoryTagger(), X=X, y=y, cv=5)

# %% In [23]:
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, performance_measure

# %% In [24]:
def to_shape(x, y):
    result = []
    start = 0
    for i in y:
        end = start + len(i)
        result.append(list(x[start:end]))
        start = end
    
    return result

# %% In [25]:
with exp.test():
    y_pred = to_shape(pred, tags)

    f1 = f1_score(tags, y_pred)
    recall = recall_score(tags, y_pred)
    precision = precision_score(tags, y_pred)

    metrics = {
        "f1": '{:04.4f}'.format(f1 * 100),
        "recall": '{:04.4f}'.format(recall * 100),
        "precision": '{:04.4f}'.format(precision * 100)
    }
    exp.log_metrics(metrics)

# %% In [26]:
print(classification_report(tags, y_pred))

# %% In [27]:
exp.end()

