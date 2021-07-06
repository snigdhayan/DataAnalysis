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


# PCA, covariance matrix and eigenvalues/eigenvectors of covariance matrix


from sklearn.decomposition import PCA
import time

time_start = time.time()

n_comp = 0.9
pca_breast_cancer = PCA(n_components=n_comp)
principal_components = pca_breast_cancer.fit_transform(X_norm,Y)

print('Time elapsed: {} seconds'.format(round(time.time()-time_start,4)))

# cov = pca_breast_cancer.get_covariance()
# eig_vals, eig_vecs = np.linalg.eig(cov)

# Prepare and export transformed dataset with principal components and labels

pc_labels = []
for i in range(0,pca_breast_cancer.singular_values_.size):
    pc_labels.append("principal component " + str(i+1)) 

principal_components_df = pd.DataFrame(data = principal_components, columns = pc_labels)
principal_components_df['label'] = Y
# principal_components_df.to_csv('pc_breast_cancer_dataset.csv',index=False)


# Visualize results

data = {"Principal component":pc_labels, 
        "Explained variance ratio":np.around(pca_breast_cancer.explained_variance_ratio_,2)}
pca_result = pd.DataFrame(data=data).sort_values(by=["Explained variance ratio"],ascending=False)
print('Explained variance ratio per principal component:\n{}\n'.format(pca_result))


# print('Tail of principal components: {}\n'.format(principal_components_df.tail()))
# print('30 eigenvalues: {}'.format(eig_vals))


# Plot the top two principal components and color according to labels


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
    indicesToKeep = breast_cancer_dataset['label'] == target
    plt.scatter(principal_components_df.loc[indicesToKeep, 'principal component 1'],
                principal_components_df.loc[indicesToKeep, 'principal component 2'], 
                c = color, 
                s = 50)

plt.legend(targets,prop={'size': 15})






