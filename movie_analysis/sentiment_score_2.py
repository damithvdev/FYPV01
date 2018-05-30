from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import csv

# read training data
with open("D:/Project/Data/sentiment_score/train.csv", 'r') as file:
    reviews = list(csv.reader(file))

with open("D:/Project/Data/sentiment_score/test3.csv", 'r') as file:
    test = list(csv.reader(file))


# Generate counts from text using a vectorizer.  There are other vectorizers available, and lots of options you can set.
# This performs our step of computing word counts.
vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
train_features = vectorizer.fit_transform([r[2] for r in reviews])
test_features = vectorizer.transform([r[2] for r in test])

# Fit a naive bayes model to the training data.
# This will train the model using the word counts we computer, and the existing classifications in the training set.
nb = MultinomialNB()
nb.fit(train_features, [int(r[3]) for r in reviews])

# Now we can use the model to predict classifications for our test features.
predictions = nb.predict(test_features)

print(predictions)

actual = [int(r[3]) for r in test]

# Compute the error.
# It is slightly different from our model because the internals of
# this process work differently from our implementation.
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)
print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))

