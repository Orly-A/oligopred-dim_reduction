
import umap.umap_ as umap
import re
import os
# import requests
# from tqdm.auto import tqdm
import pickle
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

# for an interactive plot
#%matplotlib qt




def PCA_analysis(x, n_components=3):
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(pd.DataFrame(x))
    principalDf = pd.DataFrame(data=principalComponents)
    return principalDf

def PCA_plot(principalDf, tab_test):
    finalDf = pd.concat([principalDf, tab_test[['nsub', "code"]]], axis=1)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlabel('pc1', fontsize = 15)
    ax.set_ylabel('pc2', fontsize = 15)
    ax.set_zlabel('pc3', fontsize = 15)
    ax.set_title('3_component_PCA', fontsize = 20)
    targets = set(tab_test['nsub'].to_list())
    colors = ['r', 'g', 'b']
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', "brown", 'pink', 'gray', 'purple', 'orange', 'indigo', 'teal', 'peru', 'lightgreen', 'tan', 'lime']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['nsub'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 0]
                   , finalDf.loc[indicesToKeep, 1], finalDf.loc[indicesToKeep, 2]
                   , c = color
                   , s = 20, alpha=0.2)
    ax.legend(targets)
    ax.grid()
    plt.show()

def supervised_umap_analysis(x, y, full_tab_with_clust_embed):

    s_reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=350,
                          min_dist=0.5)
    x = StandardScaler().fit_transform(x)
    s_umap_embeds = s_reducer.fit_transform(x, y)
    s_umap_embeds_350_df = pd.DataFrame(data=s_umap_embeds)
    finalDf_350 = pd.concat([s_umap_embeds_350_df, full_tab_with_clust_embed[['nsub', "code", 'representative']]], axis=1)
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/dim_reduction/s_umap_embeds_350_full_data_esm.pkl", "wb") as f:
        pickle.dump(finalDf_350, f)
    return finalDf_350


def umap_plot2(finalDf):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('dimension1', fontsize=15)
    ax.set_ylabel('dimension2', fontsize=15)
    ax.set_zlabel('dimension3', fontsize=15)
    ax.set_title('3_component_supervised_umap', fontsize = 20)
    targets = set(finalDf['nsub'].to_list())
    # colors = ['r', 'g', 'b']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'lime', "brown", 'pink', 'gray', 'purple', 'orange', 'indigo', 'teal', 'peru', 'lightgreen', 'tan', 'w']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['nsub'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 0]
                   , finalDf.loc[indicesToKeep, 1], finalDf.loc[indicesToKeep, 2]
                   , c=color
                   , s=20, alpha=0.2)
    ax.legend(targets)
    ax.grid()
    plt.show()


def plot_for_presentation(finalDf):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('dim1', fontsize=15)
    ax.set_ylabel('dim2', fontsize=15)
    ax.set_zlabel('dimn3', fontsize=15)
    # ax.set_title('3_component_supervised_umap', fontsize = 20)
    targets = set(finalDf['nsub'].to_list())
    # colors = ['r', 'g', 'b']
    # updated colors after ilanit poster
    colors = ['b', 'r', 'y', 'g', 'm', 'c', 'saddlebrown', 'lime', "midnightblue", 'pink', 'gray', 'darkred', 'orange',
              'indigo', 'teal', 'peru', 'lightgreen', 'tan', 'k']
    # colors = ['b', 'r', 'y', 'g', 'm', 'c', 'k', 'lime', "brown", 'pink', 'gray', 'purple', 'orange', 'indigo', 'teal',
    #           'peru', 'lightgreen', 'tan', 'w']
    ax.set_xlim([-10, 15])
    ax.set_ylim([-5, 15])
    ax.set_zlim([-12, 15])
    ax.set_xticks([-10, -5, 0, 5, 10, 15])
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.set_zticks([-10, -5, 0, 5, 10, 15])

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['nsub'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 0]
                   , finalDf.loc[indicesToKeep, 1], finalDf.loc[indicesToKeep, 2]
                   , c=color
                   , s=20)
    targets = [int(i) for i in targets]
    # ax.legend(targets, loc="right", framealpha=1)
    ax.grid()
    plt.show()


if __name__ == "__main__":
    # full_tab_with_clust_embed = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/clustering/re_clust_c0.3/full_tab_with_clust_embed.pkl")
    full_tab_with_clust_embed = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/esm_tab.pkl")
    full_tab_with_clust_embed.reset_index(drop=True, inplace=True)
    print(full_tab_with_clust_embed)
    x = pd.DataFrame(item for item in full_tab_with_clust_embed["esm_embeddings"])
    # principalDf = PCA_analysis(x, n_components=3)
    # PCA_plot(principalDf, full_tab_with_clust_embed)
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/dim_reduction/s_umap_embeds_350_full_data_esm.pkl", "rb") as f:
        finalDf = pickle.load(f)
    # finalDf = supervised_umap_analysis(x, full_tab_with_clust_embed["nsub"], full_tab_with_clust_embed)

    # umap_plot2(finalDf)
    plot_for_presentation(finalDf)
    print("finished")


