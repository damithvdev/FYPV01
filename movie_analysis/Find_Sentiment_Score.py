from collections import Counter
import csv
import re
from collections import Counter
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

with open("D:/Project/Data/sentiment_score/train.csv", 'r') as file:  # read training data
    reviews = list(csv.reader(file))


def get_text(review, score):
    return " ".join([r[2].lower() for r in review if r[3] == str(score)])


def count_text(text):
    words = re.split("\s+", text)
    return Counter(words)


# create corpus
ex_negative_text = get_text(reviews, 0)
negative_text = get_text(reviews, 1)
neutral_text = get_text(reviews, 2)
positive_text = get_text(reviews, 3)
ex_positive_text = get_text(reviews, 4)

# Generate word counts (dictionary)
ex_negative_counts = count_text(ex_negative_text)
negative_counts = count_text(negative_text)
neutral_counts = count_text(neutral_text)
positive_counts = count_text(positive_text)
ex_positive_counts = count_text(negative_text)


# count each classification occurring in data set
def get_y_count(score):
    return len([r for r in reviews if r[3] == str(score)])


ex_negative_review_count = get_y_count(0)
negative_review_count = get_y_count(1)
neutral_review_count = get_y_count(2)
positive_review_count = get_y_count(3)
ex_positive_review_count = get_y_count(4)

# class probabilities
prob_ex_positive = ex_positive_review_count / len(reviews)
prob_positive = positive_review_count / len(reviews)
prob_neutral = neutral_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)
prob_ex_negative = ex_negative_review_count / len(reviews)


def make_class_prediction(text, counts, class_prob, class_count):
    prediction = 1
    text_counts = Counter(re.split("\s+", text))
    for word in text_counts:
        prediction *= text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))

    return prediction * class_prob


def make_decision(text, make_class_prediction):
    neg_pre = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    pos_pre = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)
    neu_pre = make_class_prediction(text, neutral_counts, prob_neutral, neutral_review_count)
    ex_neg_pre = make_class_prediction(text, ex_negative_counts, prob_ex_negative, ex_negative_review_count)
    ex_pos_pre = make_class_prediction(text, ex_positive_counts, prob_ex_positive, ex_positive_review_count)

    total_prediction = neg_pre + pos_pre + neu_pre + ex_neg_pre + ex_pos_pre

    if neg_pre > pos_pre and neg_pre > neu_pre and neg_pre > ex_neg_pre and neg_pre > ex_pos_pre:
        return 1
    elif pos_pre > neg_pre and pos_pre > neu_pre and pos_pre > ex_neg_pre and pos_pre > ex_pos_pre:
        return 3
    elif neu_pre > pos_pre and neu_pre > neg_pre and neu_pre > ex_neg_pre and neu_pre > ex_pos_pre:
        return 2
    elif ex_neg_pre > pos_pre and ex_neg_pre > neu_pre and ex_neg_pre > neg_pre and ex_neg_pre > ex_pos_pre:
        return 0
    elif ex_pos_pre > pos_pre and ex_pos_pre > neu_pre and ex_pos_pre > neg_pre and ex_pos_pre > ex_neg_pre:
        return 4


with open("D:/Project/Data/sentiment_score/test3.csv", 'r') as file:
    test = list(csv.reader(file))

predictions = [make_decision(r[2], make_class_prediction) for r in test]

print(predictions)

actual = [int(r[3]) for r in test]

# Generate the roc curve using scikits-learn.
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve.  The closer to 1, the "better" the predictions.
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))


