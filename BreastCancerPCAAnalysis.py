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


# PCA, covariance matrix and eigenvalues/eigenvectors of covariance matrix


from sklearn.decomposition import PCA
pca_breast = PCA(n_components=0.9)
principalComponents_breast = pca_breast.fit_transform(X_norm,Y)

# cov = pca_breast.get_covariance()
# eig_vals, eig_vecs = np.linalg.eig(cov)


# Visualize top principal components

pc_labels = []
for i in range(0,pca_breast.singular_values_.size):
    pc_labels.append("principal component " + str(i+1)) 
    
principal_breast_Df = pd.DataFrame(data = principalComponents_breast, columns = pc_labels)

# Visualize results

data = {"Principal component":pc_labels, "Explained variance ratio":pca_breast.explained_variance_ratio_}
pca_data = pd.DataFrame(data=data).sort_values(by=["Explained variance ratio"],ascending=False)
print('Explained variance ratio per principal component:\n{}\n'.format(pca_data))


# print('Tail of principal components: {}\n'.format(principal_breast_Df.tail()))
# print('30 eigenvalues: {}'.format(eig_vals))


# Plot the top two principal components


import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component 1',fontsize=20)
plt.ylabel('Principal Component 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})






