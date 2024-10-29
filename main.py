import pandas as pd

# Veri setini yükleme
df = pd.read_csv('Mall_Customers.csv')



# Gerekli sütunları seçme
df = df[['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Sütun adlarını daha okunabilir hale getirme
df.columns = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore']

# Eksik verileri kontrol etme
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Yaş ve Harcama Puanı dağılımı
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='SpendingScore', data=df, hue='Gender')
plt.title('Age vs Spending Score')
plt.show()

# Yıllık Gelir ve Harcama Puanı dağılımı
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', data=df, hue='Gender')
plt.title('Annual Income vs Spending Score')
plt.show()


from sklearn.cluster import KMeans

# Yıllık Gelir ve Harcama Puanını kullanarak segmentasyon
X = df[['AnnualIncome', 'SpendingScore']]

# En uygun küme sayısını belirlemek için Dirsek Yöntemi
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 5 küme ile K-Means uygulama
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Küme sonuçlarını veri setine ekleme
df['Cluster'] = y_kmeans

# Küme merkezlerini görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='viridis', data=df, legend='full')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Her bir küme için ortalama yıllık gelir ve harcama puanı
cluster_summary = df.groupby('Cluster')[['AnnualIncome', 'SpendingScore']].mean()
print(cluster_summary)

# En yüksek harcama puanına sahip küme
max_spending_cluster = cluster_summary['SpendingScore'].idxmax()
print(f'En çok harcama yapan küme: {max_spending_cluster}')



# En yüksek harcama yapan küme (max_spending_cluster)
cluster_1_customers = df[df['Cluster'] == max_spending_cluster]

# Bu kümedeki bir müşteri profili örneği
sample_customer_profile = cluster_1_customers.sample(n=4, random_state=42)

print("Örnek Müşteri Profili:")
print(sample_customer_profile)

# Her kümedeki kişi sayısını bulma
cluster_counts = df['Cluster'].value_counts()
print(cluster_counts)






