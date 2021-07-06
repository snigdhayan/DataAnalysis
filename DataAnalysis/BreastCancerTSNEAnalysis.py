# coding: utf-8

# Gather breast cancer data

from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
breast_cancer_data = breast_cancer.data
breast_cancer_labels = breast_cancer.target


# Prepare data as pandas dataframe


import numpy as np
labels = np.reshape(breast_cancer_labels,(569,1))
final_breast_cancer_data = np.concatenate([breast_cancer_data,labels],axis=1)

import pandas as pd
breast_cancer_dataset = pd.DataFrame(final_breast_cancer_data)
features = breast_cancer.feature_names
features_labels = np.append(features,'label')
breast_cancer_dataset.columns = features_labels

# Replace 0,1 label by medical terminology (Benign = cancer false, Malignant = cancer true)

breast_cancer_dataset['label'].replace(0, 'Benign',inplace=True)
breast_cancer_dataset['label'].replace(1, 'Malignant',inplace=True)

# Standardize data by setting mean to 0 and standard deviation to 1

from sklearn.preprocessing import StandardScaler

X, Y = breast_cancer_dataset.drop(columns='label'), breast_cancer_dataset['label']
X_norm = StandardScaler().fit_transform(X)


# Apply t-SNE to dataset with fixed random_state


from sklearn.manifold import TSNE
import time

time_start = time.time()

n_comp = 2
tsne_breast_cancer = TSNE(n_components=n_comp,
                          perplexity=40,
                          n_iter=400,
                          random_state=8)
result_tsne = tsne_breast_cancer.fit_transform(X_norm)

print('Time elapsed: {} seconds'.format(round(time.time()-time_start,2)))

# Prepare the result of t-SNE as dataframe with labels

tsne_labels = []
for i in range(0,n_comp):
    tsne_labels.append("tsne_" + str(i+1)) 

result_tsne_df = pd.DataFrame(data = result_tsne, columns = tsne_labels)
result_tsne_df['label'] = Y


# Plot the result and color according to labels


import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('tsne_1',fontsize=20)
plt.ylabel('tsne_2',fontsize=20)
plt.title("t-SNE Visualization of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = breast_cancer_dataset['label'] == target
    plt.scatter(result_tsne_df.loc[indicesToKeep, 'tsne_1'],
                result_tsne_df.loc[indicesToKeep, 'tsne_2'], 
                c = color, 
                s = 50)

plt.legend(targets,prop={'size': 15})






