import os

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "":
    os.chdir("")

from utils import load_unlabeled_spam_dataset

df_train = load_unlabeled_spam_dataset()

ABSTAIN = -1
NOT_SPAM = 0
SPAM = 1

from snorkel.labeling import labeling_function

@labeling_function()
def lf_keyword_my(x):
    """Many spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if "my" in x.text.lower() else ABSTAIN

import re

@labeling_function()
def lf_regex_check_out(x):
    """Spam comments say 'check out my video', 'check it out', etc."""
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN

@labeling_function()
def lf_short_comment(x):
    """Non-spam comments are often short, such as 'cool video!'."""
    return NOT_SPAM if len(x.text.split()) < 5 else ABSTAIN

from textblob import TextBlob

@labeling_function()
def lf_textblob_polarity(x):
    """
    We use a third-party sentiment classification model, TextBlob.
    We combine this with the heuristic that non-spam comments are often positive.
    """
    return NOT_SPAM if TextBlob(x.text).sentiment.polarity > 0.3 else ABSTAIN

from snorkel.labeling import LabelModel, PandasLFApplier

# Define the set of labeling functions (LFs)
lfs = [lf_keyword_my, lf_regex_check_out, lf_short_comment, lf_textblob_polarity]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")


df_train = df_train[df_train.label != ABSTAIN]

import random

import nltk
from nltk.corpus import wordnet as wn
from snorkel.augmentation import transformation_function

nltk.download("wordnet", quiet=True)

def get_synonyms(word):
    """Get the synonyms of word from Wordnet."""
    lemmas = set().union(*[s.lemmas() for s in wn.synsets(word)])
    return list(set(l.name().lower().replace("_", " ") for l in lemmas) - {word})

@transformation_function()
def tf_replace_word_with_synonym(x):
    """Try to replace a random word with a synonym."""
    words = x.text.lower().split()
    idx = random.choice(range(len(words)))
    synonyms = get_synonyms(words[idx])
    if len(synonyms) > 0:
        x.text = " ".join(words[:idx] + [synonyms[0]] + words[idx + 1 :])
        return x

from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier

tf_policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
tf_applier = PandasTFApplier([tf_replace_word_with_synonym], tf_policy)
df_train_augmented = tf_applier.apply(df_train)

from snorkel.slicing import slicing_function

@slicing_function()
def short_link(x):
    """Return whether text matches common pattern for shortened ".ly" links."""
    return int(bool(re.search(r"\w+\.ly", x.text)))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train_text = df_train_augmented.text.tolist()
X_train = CountVectorizer(ngram_range=(1, 2)).fit_transform(train_text)

clf = LogisticRegression(solver="lbfgs")
clf.fit(X=X_train, y=df_train_augmented.label.values)
