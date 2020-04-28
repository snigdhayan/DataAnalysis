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


# Standardize data by setting mean to 0 and standard deviation to 1

from sklearn.preprocessing import StandardScaler

X, Y = breast_dataset.drop(columns='label'), breast_dataset['label']
X_norm = StandardScaler().fit_transform(X)


# Feature selection based on percentile f_regression

from sklearn.feature_selection import SelectPercentile, f_regression

selector = SelectPercentile(f_regression, percentile=20)
X_new = selector.fit_transform(X_norm, Y)

feature_support = selector.get_support()
selected_features = X.loc[:,feature_support].columns

# Export transformed dataset with reduced features as dataframe

breast_Df = pd.DataFrame(data = X_new, columns = selected_features)
# breast_Df.to_csv('breast_cancer_dataset.csv',index=False)

# Visualize results

n_selected = len(selected_features)
data = {"Feature": features, "Score": selector.scores_}
feature_score = pd.DataFrame(data=data).sort_values(by=["Score"],ascending=False)
print('\n{} Selected Features:\n{}\n'.format(n_selected,feature_score.head(n_selected)))



# Plot the top two features

import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel(feature_score.iloc[0,0],fontsize=20)
plt.ylabel(feature_score.iloc[1,0],fontsize=20)
plt.title("Feature Selection of Breast Cancer Dataset",fontsize=20)
targets = [0, 1]
legends = ['Benign', 'Malignant']
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(breast_Df.loc[indicesToKeep, feature_score.iloc[0,0]]
               , breast_Df.loc[indicesToKeep, feature_score.iloc[1,0]], c = color, s = 50)

plt.legend(legends,prop={'size': 15})






