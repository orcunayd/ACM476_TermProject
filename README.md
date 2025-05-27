# 26. Feature Selection with Random Forest Importance
print("\n--- FEATURE IMPORTANCE ---")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_clf, y_train_clf)

feature_importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("Feature Importances:")
print(feature_importances)

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

from sklearn.decomposition import PCA

# 27. PCA for Visualization and Dimensionality Reduction
# Normalize features before PCA
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X)

# PCA Transformation (2 Components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create PCA DataFrame for Visualization
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['label'] = y_clf.values

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=df_pca, palette='coolwarm', alpha=0.7)
plt.title('PCA: Players by Label (2 Principal Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Label')
plt.show()

# Explained Variance Ratio
print("\nExplained Variance Ratio by Each Component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ratio:.4f}")
print(f"  Total Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")
