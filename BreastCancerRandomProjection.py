# coding: utf-8

# Gather breast cancer data

from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
breast_data = breast.data
breast_labels = breast.target

# Prepare data as pandas dataframe

import numpy as np
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)

import pandas as pd
breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features_labels = np.append(features,'label')
breast_dataset.columns = features_labels

# Replace 0,1 label by medical terminology (Benign = cancer false, Malignant = cancer true)

breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)

# Standardize data by setting mean to 0 and standard deviation to 1

from sklearn.preprocessing import StandardScaler

X, Y = breast_dataset.drop(columns='label'), breast_dataset['label']
X_norm = StandardScaler().fit_transform(X)

# Apply random projection to dataset using sparse random matrices
# The underlying theory relies on the Johnson-Lindenstrauss lemma

from sklearn import random_projection
import time

time_start = time.time()

n_comp = 2
transformer = random_projection.SparseRandomProjection(n_components=n_comp)
result_rp = transformer.fit_transform(X_norm)

print('Time elapsed: {} seconds'.format(round(time.time()-time_start,2)))

# Prepare the result of random projection as dataframe with labels

rp_labels = []
for i in range(0,n_comp):
    rp_labels.append("Random_Projection_" + str(i+1)) 

result_rp_df = pd.DataFrame(data = result_rp, columns = rp_labels)
result_rp_df['label'] = Y


# Plot the result and color according to labels


import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Random_Projection_1',fontsize=20)
plt.ylabel('Randon_Projection_2',fontsize=20)
plt.title("Random Projection Visualization of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(result_rp_df.loc[indicesToKeep, 'Random_Projection_1'],
                result_rp_df.loc[indicesToKeep, 'Random_Projection_2'], 
                c = color, 
                s = 50)

plt.legend(targets,prop={'size': 15})

