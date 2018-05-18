import numpy as np
from collections import Counter
import warnings
import random
import pandas as pd

def k_nearest_neighbours(data, prediction, k=3):
    if len(data)>=k:
        warnings.warn('K is set to a value less than voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(prediction))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
df.drop(['id'],1, inplace=True)
data = df.astype(float).values.tolist()

random.shuffle(data)

test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = data[:-int(test_size*len(data))]
test_data = data[-int(test_size*len(data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
cnt = 0

for group in test_set:
    for each in test_set[group]:
        vote = k_nearest_neighbours(train_set, each, k=5)
        if group == vote:
            correct +=1
        cnt+=1

print('accuracy', correct/cnt)
