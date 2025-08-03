import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from rdkit.ML.Cluster import Butina

def compute_features(mols, method="ecfp", radius=2, nBits=1024):
    """
    Compute molecular features for visualization.
    
    Options:
        - 'ecfp': ECFP fingerprint
        - 'fcfp': FCFP fingerprint
        - 'both': Concatenate ECFP + FCFP
    
    Returns:
        numpy array of features
    """
    features = []
    if method == "ecfp":
        for mol in mols:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, useFeatures=False)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        return np.array(features), None
    
    elif method == "fcfp":
        for mol in mols:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, useFeatures=True)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        return np.array(features), None
    
    elif method == "both":
        ecfp_fps = []
        fcfp_fps = []
        for mol in mols:
            # Compute ECFP
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, useFeatures=False)
            ecfp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(ecfp, ecfp_arr)
            
            # Compute FCFP
            fcfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, useFeatures=True)
            fcfp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fcfp, fcfp_arr)
            
            # Combine
            ecfp_fps.append(ecfp_arr)
            fcfp_fps.append(fcfp_arr)
        
        combined = [np.concatenate([e, f]) for e, f in zip(ecfp_fps, fcfp_fps)]
        return np.array(combined), None
    
    else:
        raise ValueError("method must be 'ecfp', 'fcfp', or 'both'")

def reduce_dimensions(features, method="pca"):
    """
    Reduce dimensions for visualization using PCA, t-SNE, or UMAP.
    """
    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(features)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        coords = reducer.fit_transform(features)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(features)
    else:
        raise ValueError("Choose 'pca', 'tsne', or 'umap'")
    return coords

def butina_clustering(mols, cutoff=0.6, radius=2, nBits=1024):
    """
    Perform Butina clustering based on Tanimoto similarity.
    """
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) for mol in mols]
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    return clusters

def plot_clusters(coords, cluster_labels, title="Molecule Clusters"):
    """
    Plot molecules in 2D based on clustering results.
    """
    df = pd.DataFrame(coords, columns=["x", "y"])
    df["cluster"] = cluster_labels
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="x", y="y", hue="cluster", palette="tab10", data=df)
    plt.title(title)
    plt.show()
