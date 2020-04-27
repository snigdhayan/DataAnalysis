# coding: utf-8
# Created on Sat Apr 25 21:35:02 2020


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


# Normalize data by min and max

from sklearn.preprocessing import MinMaxScaler
# feature_columns = breast_dataset.loc[:, features].values

X, Y = breast_dataset.drop(columns='label'), breast_dataset['label']
X_norm = MinMaxScaler().fit_transform(X)


# Feature selection based on percentile chi-squared

from sklearn.feature_selection import SelectPercentile, chi2

chi_selector = SelectPercentile(chi2, percentile=20)
X_new = chi_selector.fit_transform(X_norm, Y)

chi_support = chi_selector.get_support()
selected_features = X.loc[:,chi_support].columns

breast_Df = pd.DataFrame(data = X_new, columns = selected_features)


# Visualize results

print('{} Selected Features: {}\n'.format(len(selected_features.tolist()), selected_features.tolist()))

# Plot the top two features


import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel(selected_features[0],fontsize=20)
plt.ylabel(selected_features[1],fontsize=20)
plt.title("Feature Selection of Breast Cancer Dataset",fontsize=20)
targets = [0, 1]
legends = ['Benign', 'Malignant']
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(breast_Df.loc[indicesToKeep, selected_features[0]]
               , breast_Df.loc[indicesToKeep, selected_features[1]], c = color, s = 50)

plt.legend(legends,prop={'size': 15})






