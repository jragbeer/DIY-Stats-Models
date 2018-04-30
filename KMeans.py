import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, cross_validation
import pandas as pd

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for features in X:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(features)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
print(df.head())
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for col in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:

            column_contents = df[col].values.tolist()

            uniques = set(column_contents)

            x = 0
            for each in uniques:
                if each not in text_digit_vals:

                    text_digit_vals[each] = x
                    x += 1

            df[col] = list(map(convert_to_int, df[col]))

    return df

df = handle_non_numerical_data(df)
print(df.head())

df.drop(['ticket', 'home.dest'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = K_Means()
clf.fit(X)
correct = 0

for i in range(len(X)):
    predict_values = np.array(X[i].astype(float))
    predict_values = predict_values.reshape(-1, len(predict_values))
    prediction = clf.predict(predict_values)
    if prediction == y[i]:
        correct += 1
yy= correct / len(X)
print("\nAccuracy: {:.2f}%".format(max(yy, (1-yy))*100))