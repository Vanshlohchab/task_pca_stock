# Task 7 - Part 1: PCA (Dimensionality Reduction)
# ------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA (reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Put results into a DataFrame
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["target"] = y

print("Original shape:", X.shape)
print("Reduced shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Step 5: Plot the PCA result
plt.figure(figsize=(8, 6))
colors = ["red", "green", "blue"]

for i, target_name in enumerate(target_names):
    subset = df_pca[df_pca["target"] == i]
    plt.scatter(subset["PC1"], subset["PC2"], label=target_name, color=colors[i])

plt.title("PCA of Iris Dataset (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
