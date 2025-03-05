#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.metrics import silhouette_score as sk_silhouette_score
import itertools
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import cupy as cp
from cuml.preprocessing import StandardScaler as CuStandardScaler
from cuml.cluster import KMeans as CuKMeans
from cuml.metrics.cluster import silhouette_score as cu_silhouette_score
import gc
import time
import os
import subprocess



# In[2]:


with open("vetting_playlist.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Inspect the type of data
print(type(data))


tdf = pd.json_normalize(data)
tdf.info()

tdftracks=[]
for i in tdf['tracks.items']:
    tdftracks.append(i)


# In[3]:


tdf.head()


# In[4]:


ttracks=pd.DataFrame(tdftracks)


# In[5]:


ttracks.head()


# In[6]:


ttracks=ttracks.T


# In[7]:


ttracks.head()


# In[8]:


ttracks = ttracks[0].apply(pd.Series)
ttracks.head()


# In[9]:


ttracks.info()


# In[10]:


actual_tracks = ttracks['track'].apply(pd.Series)
actual_tracks.head()


# In[11]:


actual_tracks.info()


# In[12]:


songs=pd.read_csv('tracks.csv')
songs.info()


# In[13]:


true_df= pd.merge(actual_tracks, songs, on="id", how="left")
true_df.info()
true_df.head()


# In[14]:


#Features: duration, popularity, valence, tempo, acousticness, mode, energy, genre
true_df = true_df.dropna(subset=['key'])
true_df.info()
true_df.head()


# In[15]:
"""

features = ["duration_ms_x", "popularity_x", "valence", "tempo", "acousticness", "speechiness", "energy", "explicit_x"]
X = true_df[features].copy()
for i, row in true_df.iterrows():
    med = row['explicit_x']
    if med == True:
        true_df.loc[i, 'explicitnum'] = 1
    else:
        true_df.loc[i, 'explicitnum'] = 0


# In[16]:


true_df.info()
true_df.head()


# In[17]:


true_df['synthness'] = (1 - true_df['acousticness']) * true_df['instrumentalness'] * true_df['energy']


# In[18]:


features = ["duration_ms_x", "popularity_x", "valence", "tempo", "synthness"]
X = true_df[features].copy()
scaler = SkStandardScaler()
X_scaled = scaler.fit_transform(X)
def plot_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolors="k", alpha=0.7)
    plt.title(title)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    plt.close()


# In[19]:


k_values = range(1, 27)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_values, inertia_values, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for K-Means")
plt.show()


# In[20]:


kmeans = KMeans(n_clusters=14, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

plot_clusters(X_scaled, kmeans_labels, "K-Means Clustering")


# In[21]:


dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

plot_clusters(X_scaled, dbscan_labels, "DBSCAN Clustering")


# In[22]:


gmm_values = []
k_values = range(1, 27)
for k in k_values:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    gmm_values.append(-gmm.score(X_scaled))  # Use negative log-likelihood

# Plot Elbow Curve for GMM
plt.figure(figsize=(6, 4))
plt.plot(k_values, gmm_values, marker='o', linestyle='--')
plt.xlabel("Number of Components (K)")
plt.ylabel("Negative Log-Likelihood")
plt.title("Elbow Method for Gaussian Mixture Model (GMM)")
plt.show()


# In[23]:


gmm = GaussianMixture(n_components=14, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

plot_clusters(X_scaled, gmm_labels, "Gaussian Mixture Model (GMM) Clustering")
"""

# In[24]:


file_paths = [
    'mpd.slice.0-999.json',
    'mpd.slice.99000-99999.json',
    'mpd.slice.995000-995999.json',
    'mpd.slice.996000-996999.json',
    'mpd.slice.997000-997999.json',
    'mpd.slice.998000-998999.json',
    'mpd.slice.999000-999999.json'
]
scaler = SkStandardScaler()

all_data = []
batch_size = 100000  # Process 1000 tracks at a time

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        for playlist in data['playlists']:
            playlist_name = playlist['name']
            playlist_desc = playlist.get('description', '')
            
            # Process tracks in small batches
            for i in range(0, len(playlist['tracks']), batch_size):
                batch_tracks = playlist['tracks'][i:i+batch_size]
                
                for track in batch_tracks:
                    track_id = track['track_uri'].replace("spotify:track:", "")
                    all_data.append({
                        'playlist_name': playlist_name,
                        'playlist_desc': playlist_desc,
                        'position_in_playlist': track['pos'],
                        'track_name': track['track_name'],
                        'id': track_id,  
                        'artist_name': track['artist_name'],
                        'duration_ms': track['duration_ms'],
                        'album_name': track['album_name']
                    })

xdf = pd.DataFrame(all_data)

print(xdf.head())
xdf.dropna
dmz = pd.merge(xdf, songs, on = 'id', how = 'inner')
dmz.dropna()
dmz.info()
dmz.dropna(how = 'all')
dmz['synthness'] = (1 - dmz['acousticness']) * dmz['instrumentalness'] * dmz['energy']
features =["duration_ms_x", "popularity", "valence", "tempo", "synthness", 
    "danceability", "energy", "key", "loudness", "mode", "speechiness", 
    "acousticness", "instrumentalness", "liveness", "time_signature"]
Xa= dmz[features].copy()
Xa_scaled=scaler.fit_transform(Xa)
def plot_clusters(Xa, labels, title):
    plt.scatter(Xa[:, 0,Xa], Xa[:, 1], c=labels, cmap='virdis', edgecolors='K', alpha=0.7)
    plt.title(title)
    plt.xLabel(features[0])
    plt.yLabel(features[1])
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    plt.close()



# In[ ]:

device_choice = input("Select device (1 for CPU, 2 for GPU): ")

if device_choice == '2':
    # If the user selects GPU, you may need to include any specific GPU settings or libraries
    import cupy as cp  # Ensure you have CuPy installed and set up for GPU operations
    Xa_scaled = cp.asarray(Xa_scaled)  # Transfer data to GPU
else:
    # Default to CPU; nothing changes
    Xa_scaled = np.array(Xa_scaled)


"""
k_values = range(2, 27)
inertia_values = []
silhouette_scores = []

for k in k_values:
    skmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    if device_choice == '2':
        labels = skmeans.fit_predict(Xa_scaled.get())
    else:
        labels = skmeans.fit_predict(Xa_scaled)
   
    inertia_values.append(kmeans.inertia_)
    if k + 1 > 1:
        silhouette = sk_silhouette_score(Xa_scaled.get() if device_choice == '2' else Xa_scaled, labels)
        silhouette_scores.append(silhouette)
        print(f"Clusters: {k}, Silhouette Score: {silhouette:.4f}")
    else:
        silhouette_scores.append(-1)  # Not valid for 1 cluster
        print(f"Clusters: {k}, Silhouette Score: Not applicable")
plt.figure(figsize=(6, 4))
plt.plot(k_values, inertia_values, marker='o', linestyle='--')
plt.xlabel('Number of Cluster (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for K-Means')
plt.savefig(f"elbowkmeans.png", dpi=300, bbox_inches="tight")
plt.close()
soptimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
optimal_silhouette = max(silhouette_scores)
print(f"\nOptimal Number of Clusters: {soptimal_k}")
print(f"Highest Silhouette Score: {optimal_silhouette:.4f} (Best Model)")

"""

"""
# In[ ]:
eps_values = np.arange(4, 6, 0.1)
min_samples_values = range(2, 10)

best_score = -1
best_params = None

for eps, min_samples in itertools.product(eps_values, min_samples_values):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    if device_choice == '2':
        labels = dbscan.fit_predict(Xa_scaled.get())
    else:
        Xa_scaled_limited = Xa_scaled[:150000]
        labels = dbscan.fit_predict(Xa_scaled_limited)
    
    # Count number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print(f"eps: {eps}. Min Samples: {min_samples}. Number of clusters: {n_clusters}")
    
    if n_clusters <= 1:
        print(f"  - Only {n_clusters} cluster found. Skipping silhouette calculation.")
        continue
        
    # Only calculate silhouette score when there are multiple clusters
    score = sk_silhouette_score(Xa_scaled_limited.get(), labels) if device_choice == '2' else sk_silhouette_score(Xa_scaled_limited, labels)
    print(f"  - Silhouette score: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_params = (eps, min_samples)
        best_n_clusters = n_clusters
    del dbscan
    del labels
    gc.collect()

if best_score == -1:
    print("No configuration produced multiple clusters. Try different parameter ranges.")
else:
    print(f"\nBest silhouette score: {best_score:.4f}")
    print(f"Optimal eps: {best_params[0]}, Optimal min_samples: {best_params[1]}")
    print(f"Number of clusters with optimal parameters: {best_n_clusters}")
"""




# In[ ]:
def calculate_silhouette(X, labels, model_name=""):
    if len(np.unique(labels)) >= 2 and -1 not in labels:
        score = sk_silhouette_score(X, labels)
        print(f"{model_name} Silhouette Score: {score:.4f}")
        return score
    elif len(np.unique(labels)) >= 2:  # Some points are noise (-1)
        # Calculate score only on non-noise points
        mask = labels != -1
        if np.sum(mask) > 1:  # Ensure we have at least 2 non-noise points
            score = sk_silhouette_score(X[mask], labels[labels != -1])
            print(f"{model_name} Silhouette Score (excluding noise points): {score:.4f}")
            return score
        else:
            print(f"{model_name}: Cannot calculate silhouette score - most points are noise")
            return None
    else:
        print(f"{model_name}: Cannot calculate silhouette score - only one cluster found")
        return None




# In[ ]:
# Update feature lists with the new features
true_df['synthness'] = (1 - true_df['acousticness']) * true_df['instrumentalness'] * true_df['energy']

spindt_features = [
    "duration_ms_x", "popularity_x", "valence", "tempo", "synthness", 
    "danceability", "energy", "key", "loudness", "mode", "speechiness", 
    "acousticness", "instrumentalness", "liveness", "time_signature"
]
X = true_df[spindt_features].copy()
scaler = SkStandardScaler()
X_scaled = scaler.fit_transform(X)
dmz['popularity_x'] = dmz['popularity']  # Create the popularity_x column
dmz = dmz.drop_duplicates(subset=['id'])

dmz = dmz.drop('popularity', axis=1)
# Define the equivalent feature names in your dmz dataset
dmz_feature_mapping = {
    "popularity_x": "popularity_x",
    "duration_ms_x": "duration_ms_x",
    # Add mappings for all other features
    "valence": "valence",
    "tempo": "tempo",
    "synthness": "synthness",
    "danceability": "danceability",
    "energy": "energy", 
    "key": "key",
    "loudness": "loudness",
    "mode": "mode",
    "speechiness": "speechiness",
    "acousticness": "acousticness",
    "instrumentalness": "instrumentalness",
    "liveness": "liveness",
    "time_signature": "time_signature"
}
# Create the properly aligned feature list for dmz
dmz_features = []
for feature in spindt_features:
    if feature in dmz_feature_mapping:
        mapped_feature = dmz_feature_mapping[feature]
        # Check if the mapped feature exists in dmz
        if mapped_feature in dmz.columns:
            dmz_features.append(mapped_feature)
        else:
            print(f"Warning: Mapped feature '{mapped_feature}' not found in dmz. Creating with zeros.")
            dmz[mapped_feature] = 0
            dmz_features.append(mapped_feature)
    else:
        print(f"Warning: No mapping defined for '{feature}'. Creating with zeros.")
        dmz[feature] = 0
        dmz_features.append(feature)

# Now extract the data using the properly aligned features
Xa = dmz[dmz_features].copy()
Xa_scaled = scaler.transform(Xa)

Xa_scaled = Xa_scaled[:150000]  # Limit to 150,000 points for faster processing

scaler_means = scaler.mean_
scaler_scales = scaler.scale_
# Calculate raw (unscaled) feature statistics for Spindt data
print("Calculating feature statistics for Spindt data...")
spindt_raw_stats = {}
for feature_idx, feature_name in enumerate(spindt_features):
    # Calculate using the original feature values, not scaled values
    spindt_raw_stats[feature_name] = {
        'mean': np.mean(true_df[feature_name]),
        'median': np.median(true_df[feature_name]),
        'std': np.std(true_df[feature_name]),
        'min': np.min(true_df[feature_name]),
        'max': np.max(true_df[feature_name])
    }
    print(f"Spindt {feature_name}: mean={spindt_raw_stats[feature_name]['mean']:.4f}, std={spindt_raw_stats[feature_name]['std']:.4f}")

# Calculate scaled feature statistics for comparison with Xa_scaled data
"""
# Function to compare a cluster to Spindt feature statistics
"""
def compare_cluster_to_spindt(cluster_data, spindt_stats, feature_names, scaler_means, scaler_scales):
    
   #Compare a cluster's features to the overall Spindt feature statistics.
   #Unscales the data first for proper comparison.
  # Returns similarity scores and comparisons for each feature.
    
    results = {}
    overall_similarity = 0
    
    # Unscale the cluster data
    unscaled_cluster_data = np.zeros_like(cluster_data)
    for feature_idx in range(cluster_data.shape[1]):
        # Reverse the standard scaling: X = X_scaled * scale + mean
        unscaled_cluster_data[:, feature_idx] = cluster_data[:, feature_idx] * scaler_scales[feature_idx] + scaler_means[feature_idx]
    
    for feature_idx, feature_name in enumerate(feature_names):
        # Calculate using unscaled values
        cluster_mean = np.mean(unscaled_cluster_data[:, feature_idx])
        spindt_mean = spindt_stats[feature_name]['mean']
        spindt_std = spindt_stats[feature_name]['std']
        
        # Z-score of cluster mean relative to Spindt distribution
        z_score = (cluster_mean - spindt_mean) / (spindt_std + 1e-10)  # Avoid division by zero
        
        # Convert to similarity score (1.0 = identical, 0.0 = very different)
        # Using a bell curve function where similarity decreases as |z_score| increases
        similarity = np.exp(-(z_score**2) / 2)
        
        # Determine if feature value is significantly higher/lower than Spindt average
        if z_score > 1.0:
            comparison = "significantly higher"
        elif z_score < -1.0:
            comparison = "significantly lower"
        else:
            comparison = "similar"
            
        results[feature_name] = {
            'cluster_mean': cluster_mean,
            'spindt_mean': spindt_mean,
            'z_score': z_score,
            'similarity': similarity,
            'comparison': comparison
        }
        
        overall_similarity += similarity
    
    # Average similarity across all features
    overall_similarity /= len(feature_names)
    
    return overall_similarity, results

# Fit the PRIMARY and SECONDARY clustering models (keeping these as DBSCAN)
PRIMARYDBSCAN = DBSCAN(eps=4.3, min_samples=3)
print("Fitting PRIMARYDBSCAN model...")
PLabels = PRIMARYDBSCAN.fit_predict(Xa_scaled)
# Calculate silhouette scores for each model
print("\nCalculating silhouette Primary scores:")
PScore = calculate_silhouette(Xa_scaled, PLabels, "PRIMARYDBSCAN")
p_unique_clusters = len(np.unique([l for l in PLabels if l != -1]))
print(f"PRIMARYDBSCAN: Total points={len(PLabels)}, Non-noise points={np.sum(PLabels != -1)}, Unique clusters={p_unique_clusters}")
if p_unique_clusters > 0:
    print("\nComparing PRIMARYDBSCAN clusters with Spindt feature statistics:")
    print("=" * 80)
    
    for cluster_id in np.unique([l for l in PLabels if l != -1]):
        cluster_mask = PLabels == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_data = Xa_scaled[cluster_mask]
        
        overall_sim, feature_comparisons = compare_cluster_to_spindt(
            cluster_data, spindt_raw_stats, spindt_features, scaler_means, scaler_scales
        )
        
        print(f"\nPRIMARYDBSCAN Cluster {cluster_id} (size: {cluster_size} points)")
        print(f"Overall similarity to Spindt: {overall_sim:.4f}")
        print("Feature-by-feature comparison:")
        
        # Sort features by most distinctive (lowest similarity)
        sorted_features = sorted(feature_comparisons.items(), key=lambda x: x[1]['similarity'])
        
        for feature_name, stats in sorted_features:
            print(f"  {feature_name}: {stats['comparison']} (z-score: {stats['z_score']:.2f}, similarity: {stats['similarity']:.4f})")
            print(f"    Cluster mean: {stats['cluster_mean']:.4f}, Spindt mean: {stats['spindt_mean']:.4f}")
else:
    print("\nNo PRIMARY clusters to compare with Spindt feature statistics.")

del PLabels
del PRIMARYDBSCAN
gc.collect()


SecondaryDBSCAN = DBSCAN(eps=4.4, min_samples=6) 

print("Fitting SecondaryDBSCAN model...")
SLabels = SecondaryDBSCAN.fit_predict(Xa_scaled)


print("\nCalculating silhouette Secondary scores:")
SScore = calculate_silhouette(Xa_scaled, SLabels, "SecondaryDBSCAN")

# Print cluster statistics
print("\nClustering Statistics:")

s_unique_clusters = len(np.unique([l for l in SLabels if l != -1]))

print(f"SecondaryDBSCAN: Total points={len(SLabels)}, Non-noise points={np.sum(SLabels != -1)}, Unique clusters={s_unique_clusters}")

# Compare PRIMARY clusters with Spindt feature statistics


# Compare SECONDARY clusters with Spindt feature statistics
if s_unique_clusters > 0:
    print("\nComparing SecondaryDBSCAN clusters with Spindt feature statistics:")
    print("=" * 80)
    
    for cluster_id in np.unique([l for l in SLabels if l != -1]):
        cluster_mask = SLabels == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_data = Xa_scaled[cluster_mask]
        
        overall_sim, feature_comparisons = compare_cluster_to_spindt(
    cluster_data, spindt_raw_stats, spindt_features, scaler_means, scaler_scales
)

        
        print(f"\nSecondaryDBSCAN Cluster {cluster_id} (size: {cluster_size} points)")
        print(f"Overall similarity to Spindt: {overall_sim:.4f}")
        print("Feature-by-feature comparison:")
        
        # Sort features by most distinctive (lowest similarity)
        sorted_features = sorted(feature_comparisons.items(), key=lambda x: x[1]['similarity'])
        
        for feature_name, stats in sorted_features:
            print(f"  {feature_name}: {stats['comparison']} (z-score: {stats['z_score']:.2f}, similarity: {stats['similarity']:.4f})")
            print(f"    Cluster mean: {stats['cluster_mean']:.4f}, Spindt mean: {stats['spindt_mean']:.4f}")
else:
    print("\nNo SECONDARY clusters to compare with Spindt feature statistics.")
