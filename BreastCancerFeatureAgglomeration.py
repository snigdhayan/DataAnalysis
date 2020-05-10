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

"""
Replace 0,1 label by medical terminology (Benign = cancer false, Malignant = cancer true)

breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)
"""

# Standardize data by setting mean to 0 and standard deviation to 1

from sklearn.preprocessing import StandardScaler

X, Y = breast_dataset.drop(columns='label'), breast_dataset['label']
X_norm = StandardScaler().fit_transform(X)


# Feature agglomeration based on 'euclidean' affinity and 'ward' linkage

from sklearn import cluster

agglomerate = cluster.FeatureAgglomeration(n_clusters=None,
                                           linkage="ward",
                                           affinity="euclidean",
                                           compute_full_tree=True,
                                           distance_threshold=0.2)
X_transformed = agglomerate.fit_transform(X_norm, Y)

# X_restored = pd.DataFrame(agglomerate.inverse_transform(X_transformed))

# Define the plot_dendrogram function

from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Draw the top two levels of the dendrogram of agglomerated features

import matplotlib.pyplot as plt

plot_dendrogram(agglomerate, truncate_mode='level', p=2)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel("Number of points in node (or index of point if no parenthesis).")



