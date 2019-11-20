# Writing Labeling Functions
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1
from snorkel.labeling.lf import labeling_function

book_to_first_author = dict(zip(df_books.book_idx, df_books.first_author))
first_author_to_books_df = df_books.groupby("first_author")[["book_idx"]].agg(set)
first_author_to_books = dict(
    zip(first_author_to_books_df.index, first_author_to_books_df.book_idx)
)


@labeling_function(
    resources=dict(
        book_to_first_author=book_to_first_author,
        first_author_to_books=first_author_to_books,
    )
)
def shared_first_author(x, book_to_first_author, first_author_to_books):
    author = book_to_first_author[x.book_idx]
    same_author_books = first_author_to_books[author]
    num_read = len(set(x.book_idxs).intersection(same_author_books))
    return POSITIVE if num_read > 15 else ABSTAIN
We can also leverage the long text reviews written by users to guess whether they liked or disliked a book. For example, the third df_dev entry above has a review with the text '4.5 STARS', which indicates that the user liked the book. We write a simple LF that looks for similar phrases to guess the user’s rating of a book. We interpret >= 4 stars to indicate a positive rating, while < 4 stars is negative.

low_rating_strs = [
    "one star",
    "1 star",
    "two star",
    "2 star",
    "3 star",
    "three star",
    "3.5 star",
    "2.5 star",
    "1 out of 5",
    "2 out of 5",
    "3 out of 5",
]
high_rating_strs = ["5 stars", "five stars", "four stars", "4 stars", "4.5 stars"]


@labeling_function(
    resources=dict(low_rating_strs=low_rating_strs, high_rating_strs=high_rating_strs)
)
def stars_in_review(x, low_rating_strs, high_rating_strs):
    if not isinstance(x.review_text, str):
        return ABSTAIN
    for low_rating_str in low_rating_strs:
        if low_rating_str in x.review_text.lower():
            return NEGATIVE
    for high_rating_str in high_rating_strs:
        if high_rating_str in x.review_text.lower():
            return POSITIVE
    return ABSTAIN
We can also run TextBlob, a tool that provides a pretrained sentiment analyzer, on the reviews, and use its polarity and subjectivity scores to estimate the user’s rating for the book. As usual, these thresholds were picked by analyzing the score distributions and running error analysis.

from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_polarity(x):
    if isinstance(x.review_text, str):
        x.blob = TextBlob(x.review_text)
    else:
        x.blob = None
    return x


# Label high polarity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def polarity_positive(x):
    if x.blob:
        if x.blob.polarity > 0.3:
            return POSITIVE
    return ABSTAIN


# Label high subjectivity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def subjectivity_positive(x):
    if x.blob:
        if x.blob.subjectivity > 0.75:
            return POSITIVE
    return ABSTAIN


# Label low polarity reviews as negative.
@labeling_function(pre=[textblob_polarity])
def polarity_negative(x):
    if x.blob:
        if x.blob.polarity < 0.0:
            return NEGATIVE
    return ABSTAIN
from snorkel.labeling import PandasLFApplier, LFAnalysis

lfs = [
    stars_in_review,
    shared_first_author,
    polarity_positive,
    subjectivity_positive,
    polarity_negative,
]

applier = PandasLFApplier(lfs)
L_dev = applier.apply(df_dev)
LFAnalysis(L_dev, lfs).lf_summary(df_dev.rating)
