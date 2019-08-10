# %% In [1]:
!pip install comet_ml
!pip install seqeval[gpu]

# %% In [2]:
# Comet Experiment Setup
from comet_ml import Experiment

# %% In [3]:
model_name = 'Bi-LSTM'
version = 2
dataset_name = 'citation'
exp_name = f'{model_name}_v{version}'

# %% In [4]:
%%writefile .env
COMET_API_KEY=<HIDDEN>

# %% In [5]:
exp = Experiment(project_name="ner-citation-model")
exp.set_name(exp_name)

# %% In [6]:
# Dataset
import pandas as pd
import numpy as np

# %% In [7]:
data = pd.read_csv(f"../input/ner-lists-87_citations-1000_{dataset_name}.csv")
data = data.fillna(method="ffill")
data.tail(10)

# %% In [8]:
words = list(set(data["word"].values))
words.append("PAD")
n_words = len(words)
print("# words:", n_words)

# %% In [9]:
tags = list(set(data["tag"].values))
tags.append('PAD')
n_tags = len(tags)
print('# tags:', n_tags)

# %% In [10]:
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

# %% In [11]:
getter = SentenceGetter(data)

# %% In [12]:
sentences = getter.sentences

# %% In [13]:
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# %% In [14]:
plt.hist([len(s) for s in sentences], bins=50)
plt.xlabel('Sentence Length')
plt.ylabel('No. Sentences')
exp.log_figure('Sentence length distribution', plt)

# %% In [15]:
max_len = 150
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: t for t, i in word2idx.items()}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

# %% In [16]:
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# %% In [17]:
X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

# %% In [18]:
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["PAD"])
y = np.array([to_categorical(i, num_classes=n_tags) for i in y])

# %% In [19]:
from sklearn.model_selection import train_test_split

# %% In [20]:
random_state = 45
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)
print('No. Training dataset:', X_train.shape[0])
print('No. Validation dataset:', X_valid.shape[0])
print('No. Test dataset:', X_test.shape[0])

# %% In [21]:
# Model Definitaion -> Bi-LSTM
from keras import backend as K
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report, precision_score, recall_score

# %% In [22]:
params = {
    #dataset
    'n_train': X_train.shape[0],
    
    # sizes hps
    'vocab_size': n_words,
    'max_len': max_len,
    'num_classes': n_tags,
    'embedding_size': 140,
    
    # models hps
    'optimizer': 'adam',
    'model_type': model_name,
    'layer_1_units': 100,
    'dropout': 0.1,
    'recurrent_dropout': 0.1,
    
    # training hps
    'batch_size': 32,
    'epochs': 30
}

exp.log_parameters(params)

# %% In [23]:
input = Input(shape=(max_len,))
model = Embedding(input_dim=params['vocab_size'], output_dim=params['embedding_size'], input_length=params['max_len'])(input)
model = Dropout(params['dropout'])(model)
model = Bidirectional(LSTM(units=params['layer_1_units'], return_sequences=True, recurrent_dropout=params['recurrent_dropout']))(model)
out = TimeDistributed(Dense(params['num_classes'], activation="softmax"))(model)  # softmax output layer

# %% In [24]:
model = Model(input, out)
model.summary()

# %% In [25]:
model.compile(optimizer=params['optimizer'], loss="categorical_crossentropy")

# %% In [26]:
# Training
class F1Metrics(Callback):

    def __init__(self, id2label, pad_value=0, validation_data=None, digits=4):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
            digits (int or None): number of digits in printed classification report
              (use None to print only F1 score without a report).
        """
        super(F1Metrics, self).__init__()
        self.id2label = id2label
        self.pad_value = pad_value
        self.validation_data = validation_data
        self.digits = digits
        self.is_fit = validation_data is None

    def convert_idx_to_name(self, y, array_indexes):
        """Convert label index to name.
        Args:
            y (np.ndarray): label index 2d array.
            array_indexes (list): list of valid index arrays for each row.
        Returns:
            y: label name list.
        """
        y = [[self.id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    def predict(self, X, y):
        """Predict sequences.
        Args:
            X (np.ndarray): input data.
            y (np.ndarray): tags.
        Returns:
            y_true: true sequences.
            y_pred: predicted sequences.
        """
        y_pred = self.model.predict_on_batch(X)

        # reduce dimension.
        y_true = np.argmax(y, -1)
        y_pred = np.argmax(y_pred, -1)

        non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0] for y_true_row in y_true]

        y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
        y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)

        return y_true, y_pred

    def score(self, y_true, y_pred):
        """Calculate f1 score.
        Args:
            y_true (list): true sequences.
            y_pred (list): predicted sequences.
        Returns:
            score: f1 score.
        """
        score = f1_score(y_true, y_pred)
        print(' - valid_f1: {:04.2f}'.format(score * 100))
        return score

    def on_epoch_end(self, epoch, logs={}):
        if self.is_fit:
            self.on_epoch_end_fit(epoch, logs)
        else:
            self.on_epoch_end_fit_generator(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        X = self.validation_data[0]
        y = self.validation_data[1]
        y_true, y_pred = self.predict(X, y)
        score = self.score(y_true, y_pred)
        logs['valid_f1'] = score

    def on_epoch_end_fit_generator(self, epoch, logs={}):
        y_true = []
        y_pred = []
        for X, y in self.validation_data:
            y_true_batch, y_pred_batch = self.predict(X, y)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
        score = self.score(y_true, y_pred)
        logs['valid_f1'] = score

# %% In [27]:
f1_metrics = F1Metrics(idx2tag, tag2idx['PAD'])

# %% In [28]:
with exp.train():
    history = model.fit(X_train, y_train, 
                        batch_size=params['batch_size'], 
                        epochs=params['epochs'], 
                        validation_data=(X_valid, y_valid),
                        callbacks=[f1_metrics],
                        verbose=1)

# %% In [29]:
# Evaluate
def to_char(x, y, pred):
    for x, y, p in zip([idx2word[x] for x in x], [idx2tag[x] for x in y], [idx2tag[x] for x in pred]):
        print(x, y, p)

def evaluate(X_test, y_test):
    y_pred = model.predict(X_test)
    
    # reduce dimension.
    y_true = np.argmax(y_test, -1)
    y_pred = np.argmax(y_pred, -1)
    
    #to_char(X_test[1], y_true[1], y_pred[1]) 
    
    # remove PAD labels
    non_pad_indexes = [np.nonzero(y_true_row != tag2idx['PAD'])[0] for y_true_row in y_true]
    y_true = f1_metrics.convert_idx_to_name(y_true, non_pad_indexes)
    y_pred = f1_metrics.convert_idx_to_name(y_pred, non_pad_indexes)
    
    # compute f1 score
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    return f1, precision, recall

# %% In [30]:
with exp.test():
    f1, precision, recall = evaluate(X_test, y_test)
    metrics = {
        'f1': '{:04.2f}'.format(f1 * 100),
        'precision': '{:04.2f}'.format(precision * 100),
        'recall': '{:04.2f}'.format(recall * 100),
    }
    exp.log_metrics(metrics)

# %% In [31]:
exp.end()

