import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# 1. Read the dataset
df = pd.read_csv(r"C:\Users\Orcun\OneDrive\Desktop\player_stats.csv", index_col=0)

# 2. Display the first 20 rows
print("First 20 Rows:")
print(df.head(20))

# 3. Identify numerical variables
num_var = [col for col in df.columns if df[col].dtype != 'O']
print("\nNumerical Variables:", num_var)

# 4. Descriptive statistics of numerical variables
print("\nDescriptive Statistics for Numerical Variables:")
print(df[num_var].describe().T)

# 5. General information about the DataFrame
print("\nDataset Shape:", df.shape)
print("\nDataFrame Info:")
print(df.info())

# 6. Column names
print("\nColumns:", df.columns)

# 7. Check for missing values
print("\nAny Missing Values?:", df.isnull().values.any())

# 8. Correlation matrix heatmap
corr = df[num_var].corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap='RdBu', annot=True)
plt.title("Correlation Heatmap of Numerical Variables")
plt.show()

# 9. Summary function for univariate analysis
def num_summary(dataframe, numerical_col, plot=True):
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(f"\nStatistics for {numerical_col}:")
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} Histogram")
        plt.show()

# 10. Summary for all numerical variables
for col in num_var:
    num_summary(df, col, plot=True)

# 11. Player counts by country
print("\nPlayer Counts by Country:")
country_counts = df['country'].value_counts()
print(country_counts)

plt.figure(figsize=(12,6))
sns.barplot(x=country_counts.index[:10], y=country_counts.values[:10])
plt.xticks(rotation=45)
plt.title('Top 10 Countries by Player Count')
plt.xlabel('Country')
plt.ylabel('Number of Players')
plt.show()

# 12. Top 10 players by rating
print("\nTop 10 Players by Rating:")
top10_rating = df.sort_values(by='rating', ascending=False).head(10)
print(top10_rating[['name', 'rating']])

# 13. Top 10 players by KD ratio
print("\nTop 10 Players by Kill/Death (KD) Ratio:")
top10_kd = df.sort_values(by='kd', ascending=False).head(10)
print(top10_kd[['name', 'kd']])

# 14. Top 10 players by total maps played
print("\nTop 10 Players by Total Maps Played:")
top10_maps = df.sort_values(by='total_maps', ascending=False).head(10)
print(top10_maps[['name', 'total_maps']])

# 15. Boxplot for Rating
plt.figure(figsize=(8,6))
sns.boxplot(x=df['rating'])
plt.title('Rating Distribution (Boxplot)')
plt.show()

# 16. Boxplot for each numerical variable
for col in num_var:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[col])
    plt.title(f'{col} Distribution (Boxplot)')
    plt.show()

# 17. Classify players by mean rating
mean_rating = df['rating'].mean()
good_players = df[df['rating'] > mean_rating]
bad_players = df[df['rating'] <= mean_rating]

df['label'] = (df['rating'] > mean_rating).astype(int)

good_players = df[df['label'] == 1]
bad_players = df[df['label'] == 0]

print("\nMean Rating:", mean_rating)
print("\nNumber of Good Players:", good_players.shape[0])
print("Number of Bad Players:", bad_players.shape[0])

# 18. Plot good vs bad players
plt.figure(figsize=(6,6))
sns.barplot(x=['Good Players', 'Bad Players'], y=[good_players.shape[0], bad_players.shape[0]])
plt.title('Number of Good vs Bad Players')
plt.ylabel('Player Count')
plt.show()

# 19. Country average rating
print("\nAverage Rating by Country:")
country_rating_means = df.groupby('country')['rating'].mean().sort_values(ascending=False)
print(country_rating_means.head(10))

plt.figure(figsize=(12,6))
sns.barplot(x=country_rating_means.index[:10], y=country_rating_means.values[:10])
plt.xticks(rotation=45)
plt.title('Top 10 Countries by Average Rating')
plt.xlabel('Country')
plt.ylabel('Average Rating')
plt.show()

# 20. Hierarchical Clustering and Dendrogram
# Drop categorical and missing values for clustering
df_cluster = df[num_var].dropna()

# Standardize the data before clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

# Compute linkage matrix
linkage_matrix = linkage(df_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(15, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
plt.title("Hierarchical Clustering Dendrogram (Top 30 Leaves)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# 22. Classification Preparation
X = df[num_var].drop('rating', axis=1)
y_class = df['label']

# Train-test split classification 
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)

scaler_cls = StandardScaler()
X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
X_test_cls_scaled = scaler_cls.transform(X_test_cls)

# 23. Classification Model: Logistic Regression
model_cls = LogisticRegression()
model_cls.fit(X_train_cls_scaled, y_train_cls)
y_pred_cls = model_cls.predict(X_test_cls_scaled)

print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test_cls, y_pred_cls))


# 24. Regression Preparation
y_reg = df['rating']  # rating

# Train-test split regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# 25. Regression Model: Linear Regression
model_reg = LinearRegression()
model_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg_scaled)

# 26. Regression Performance Metrics
r2 = r2_score(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)

print("\nRegression Performance (Linear Regression):")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Ratings (Linear Regression)")
plt.show()

# 27. Feature Selection: Random Forest Feature Importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_cls_scaled, y_train_cls)

importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nImportant Features (Random Forest):")
print(feature_importance_df.head(4))

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 28. PCA: Principal Component Analysis for Classification
from sklearn.decomposition import PCA

# I scaled the entire dataset before applying PCA
pca = PCA()
X_pca = pca.fit_transform(X_train_cls_scaled)

# I visualized the explained variance ratios
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_var), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# I classified the data using the components I obtained
pca_n = PCA(n_components=4)
X_train_pca = pca_n.fit_transform(X_train_cls_scaled)
X_test_pca = pca_n.transform(X_test_cls_scaled)

# Classification after PCA using Logistic Regression.
model_pca_cls = LogisticRegression()
model_pca_cls.fit(X_train_pca, y_train_cls)
y_pred_pca = model_pca_cls.predict(X_test_pca)

print("\nClassification Report after PCA (n_components=5):")
print(classification_report(y_test_cls, y_pred_pca))




