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

# 21. Create imbalance dataset without altering original df
df['label'] = df['rating'].apply(lambda x: 1 if x >= mean_rating else 0)

good_players = df[df['label'] == 1]
bad_players = df[df['label'] == 0]

target_bad = int(len(good_players) * 0.428)  # yaklaşık %30 kötü oyuncu oranı için

bad_sampled = bad_players.sample(n=target_bad, random_state=42)

df_imbalance = pd.concat([good_players, bad_sampled]).reset_index(drop=True)

print("\nClass Distribution After Imbalance (%):")
print(df_imbalance['label'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6,4))
sns.countplot(data=df_imbalance, x='label')
plt.title("Class Distribution (Players Rating)")
plt.xlabel("Label (0= Bad Players, 1= Good Players)")
plt.ylabel("Count")
plt.show()

# 22. Veri bölme
features = ['total_maps', 'total_rounds', 'kd_diff', 'kd']
X = df_imbalance[features]
y_reg = df_imbalance['rating']
y_clf = df_imbalance['label']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# 23. Regression Modelleri
print("\n--- REGRESSION RESULTS ---")
reg_models = {
    "Linear Regression": LinearRegression(),
    "KNN Regression (k=5)": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

for name, model in reg_models.items():
    model.fit(X_train_reg, y_train_reg)
    preds = model.predict(X_test_reg)
    print(f"{name}:")
    print(f"  R^2 Score: {r2_score(y_test_reg, preds):.4f}")
    print(f"  MSE: {mean_squared_error(y_test_reg, preds):.4f}\n")

# KNN Regression - En iyi k değeri arama
print("Best K for KNN Regression:")
def best_k_knn(X_train, X_test, y_train, y_test):
    best_k = 1
    best_score = -np.inf
    for k in range(1, 51):
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
        if score > best_score:
            best_score = score
            best_k = k
    print(f"  Best K: {best_k} with R^2: {best_score:.4f}")

best_k_knn(X_train_reg, X_test_reg, y_train_reg, y_test_reg)

# 24. Classification Modelleri
print("\n--- CLASSIFICATION RESULTS ---")
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN Classifier (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}


for name, model in clf_models.items():
    model.fit(X_train_clf, y_train_clf)
    y_pred = model.predict(X_test_clf)
    print(f"{name}:")
    print(f"  Accuracy: {accuracy_score(y_test_clf, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test_clf, y_pred):.4f}")
    print(f"  Recall: {recall_score(y_test_clf, y_pred):.4f}")
    print(classification_report(y_test_clf, y_pred))

# 25. 10-Fold Cross Validation
print("\n--- CROSS-VALIDATION RESULTS ---")
cv = KFold(n_splits=10, shuffle=True, random_state=42)
for name, model in {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}.items():
    scores = cross_val_score(model, X, y_clf, cv=cv, scoring='accuracy')
    print(f"{name} CV Accuracy: Mean={scores.mean():.4f}, Scores={scores.round(4)}")
