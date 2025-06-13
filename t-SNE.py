import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

csv_path = "C:/Users/yanch/Desktop/project/SpotifyFeatures.csv"
df = pd.read_csv(csv_path).dropna()

sampled_df = df.sample(n=2000, random_state=42)

feature_df = sampled_df.select_dtypes(include=[np.number])
meta_df = sampled_df.drop(columns=feature_df.columns)
genres = meta_df["genre"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df)

X_pca = PCA(n_components=10, random_state=42).fit_transform(X_scaled)

X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42).fit_transform(X_pca)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=pd.factorize(genres)[0], cmap='tab10', alpha=0.6)
plt.title("t-SNE of Spotify Song Features (Colored by Genre)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(scatter, label="Genre Index")
plt.tight_layout()
plt.show()
