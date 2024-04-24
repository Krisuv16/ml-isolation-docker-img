# Your updated code with additional features

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv("data.csv", on_bad_lines='skip', delimiter='|')

# Handle missing values and convert to numeric
dataset[['popularity', 'star_count', 'pull_count']] = dataset[['popularity', 'star_count', 'pull_count']].apply(pd.to_numeric, errors='coerce')
dataset[['popularity', 'star_count', 'pull_count']] = dataset[['popularity', 'star_count', 'pull_count']].fillna(dataset[['popularity', 'star_count', 'pull_count']].median())

# Handle missing values for new categorical features
dataset['categories'] = dataset['categories'].fillna('')
dataset['operating_systems'] = dataset['operating_systems'].fillna('')

# Feature scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(dataset[['popularity', 'star_count', 'pull_count']])

# Train Isolation Forest
model = IsolationForest(contamination=0.05)
model.fit(features_scaled)

# Predict anomaly scores
anomaly_scores = model.decision_function(features_scaled)
anomalies = dataset[model.predict(features_scaled) == -1]

# Visualize anomalies with marker size based on anomaly score
plt.scatter(dataset['popularity'], dataset['star_count'], s=100*(anomaly_scores - min(anomaly_scores)) / (max(anomaly_scores) - min(anomaly_scores)), c='blue', label='Normal', alpha=0.5)
plt.scatter(anomalies['popularity'], anomalies['star_count'], s=200*(anomaly_scores[model.predict(features_scaled) == -1] - min(anomaly_scores)) / (max(anomaly_scores) - min(anomaly_scores)), c='red', label='Anomaly', alpha=0.8)
plt.xlabel('Popularity')
plt.ylabel('Star Count')
plt.title('Anomaly Detection')
plt.colorbar(label='Anomaly Score')
plt.legend()

# Add interactivity
plt.gca().set_facecolor('lightgrey')
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_axisbelow(True)

# Add multiple plots for different feature pairs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(dataset['popularity'], dataset['pull_count'], c='blue', label='Normal', alpha=0.5)
plt.scatter(anomalies['popularity'], anomalies['pull_count'], c='red', label='Anomaly', alpha=0.8)
plt.xlabel('Popularity')
plt.ylabel('Pull Count')
plt.title('Popularity vs. Pull Count')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(dataset['star_count'], dataset['pull_count'], c='blue', label='Normal', alpha=0.5)
plt.scatter(anomalies['star_count'], anomalies['pull_count'], c='red', label='Anomaly', alpha=0.8)
plt.xlabel('Star Count')
plt.ylabel('Pull Count')
plt.title('Star Count vs. Pull Count')
plt.legend()

plt.tight_layout()
plt.show()

# Save anomalies to file
anomalies.to_csv("detected_anomalies.csv", index=False)

print("Detected anomalies:")
print(anomalies)
