from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from logger import info


def _save_plot(fig):
    """将 matplotlib 图保存到内存中"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    return buffer


class DataAnalysis:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.scaled_data = None
        self.cluster_labels = None
        self.cluster_summary = None
        self.scaler = StandardScaler()

        # 保存图像的属性
        self.correlation_plot = None
        self.clustering_diagnostics_plot = None
        self.cluster_distribution_plot = None
        self.cluster_centroids_plot = None
        self.cluster_means_plot = None

    def preview_data(self):
        info('####################数据预览####################')
        info(f'训练集纬度：{self.data.shape}')
        info("数据信息：")
        print(self.data.info())
        info(f'缺失值情况：\n{self.data.isna().sum()}')

    def clean_data(self):
        info('####################数据清洗####################')

        # 删除ID列
        if 'UserID' in self.data.columns:
            info('删除ID列')
            self.data.drop(['UserID'], axis=1, inplace=True)

        # 处理缺失值
        info('处理缺失值')
        for column in self.data.columns:
            if self.data[column].isna().sum() > 0:
                if self.data[column].dtype in ['float64', 'int64']:
                    median_value = self.data[column].median()
                    info(f'数值列 {column} 缺失值将被填充为中位数: {median_value}')
                    self.data[column].fillna(median_value, inplace=True)
                else:
                    mode_value = self.data[column].mode()[0]
                    info(f'分类列 {column} 缺失值将被填充为众数: {mode_value}')
                    self.data[column].fillna(mode_value, inplace=True)

        # 统一处理 Yes/No 字段
        yes_no_columns = [col for col in self.data.columns if self.data[col].dropna().isin(['Yes', 'No']).all()]
        if yes_no_columns:
            info(f'检测到以下 Yes/No 列：{yes_no_columns}')
            for column in yes_no_columns:
                self.data[column] = self.data[column].map({'Yes': 1, 'No': 0})

        # 编码分类变量
        info('编码分类变量')
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        if categorical_columns.any():
            info(f'检测到以下分类列：{list(categorical_columns)}')
            label_encoder = LabelEncoder()
            for column in categorical_columns:
                self.data[column] = label_encoder.fit_transform(self.data[column])
                info(f'分类列 {column} 已转换为数字编码')

        # 删除重复值
        duplicate_count = self.data.duplicated().sum()
        if duplicate_count > 0:
            info(f'检测到 {duplicate_count} 条重复记录，将被删除')
            self.data.drop_duplicates(inplace=True)

        # 标记和处理异常值
        info('检测并处理异常值')
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:

            if column == 'Taken_product':
                continue

            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((self.data[column] < lower_bound) | (self.data[column] > upper_bound)).sum()
            if outliers > 0:
                info(f'数值列 {column} 检测到 {outliers} 个异常值，将使用上下限进行裁剪')
                self.data[column] = self.data[column].clip(lower=lower_bound, upper=upper_bound)

        print(self.data.head())
        info('数据清洗完成！')

    def analyze_correlation(self):
        info('####################相关性分析####################')
        info('计算相关系数矩阵')
        correlation_matrix = self.data.corr()
        correlation_with_target = correlation_matrix['Taken_product'].sort_values(ascending=False)
        info(f"与目标变量相关性最高的特征：\n{correlation_with_target}")
        self._visualize_correlation(correlation_matrix)

    def perform_clustering(self, k_range=range(2, 11), optimal_k=4):
        info('####################聚类分析####################')
        info('正在对数据进行标准化')
        self.scaled_data = self.scaler.fit_transform(self.data)
        info('正在确定最佳聚类数')
        inertia = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=10).fit(self.scaled_data)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, kmeans.labels_))

        self._visualize_clustering_diagnostics(k_range, inertia, silhouette_scores)

        # 使用所选 k 的最终聚类
        info('正在进行K-均值聚类')
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=15)
        kmeans_final.fit(self.scaled_data)
        self.cluster_labels = kmeans_final.labels_

        # 添加聚类标签到原始数据
        info('正在将聚类标签添加到原始数据中以进行分析')
        self.data['Cluster'] = self.cluster_labels

        # 统计每个聚类的均值
        info('正在统计每个聚类的均值')
        self.cluster_summary = self.data.groupby('Cluster').mean()
        info('聚类统计结果：')
        print(self.cluster_summary)

        # 可视化结果
        self._visualize_clusters(kmeans_final.cluster_centers_)

    def _visualize_correlation(self, correlation_matrix):
        info('正在绘制相关性热力图')
        fig, ax = plt.subplots(figsize=(20, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f", ax=ax)
        plt.title("Correlation Matrix of Variables")
        self.correlation_plot = _save_plot(fig)

    def _visualize_clustering_diagnostics(self, k_range, inertia, silhouette_scores):
        info('正在绘制聚类诊断图')
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(k_range, inertia, marker='o')
        axes[0].set_xlabel('Number of clusters')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method For Optimal k')

        axes[1].plot(k_range, silhouette_scores, marker='o')
        axes[1].set_xlabel('Number of clusters')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score For Each k')

        plt.tight_layout()
        self.clustering_diagnostics_plot = _save_plot(fig)

    def _visualize_clusters(self, cluster_centers):
        info('正在绘制聚类结果散点图')
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            self.scaled_data[:, 0], self.scaled_data[:, 1],
            c=self.cluster_labels, cmap='viridis', s=50, alpha=0.7
        )
        plt.title("Cluster Distribution")
        plt.xlabel("Feature 1 (scaled)")
        plt.ylabel("Feature 2")
        plt.colorbar(scatter, label='Cluster Label')
        cluster_distribution_plot = _save_plot(fig)

        info('正在绘制聚类中心')
        fig, ax = plt.subplots(figsize=(8, 6))
        original_centers = self.scaler.inverse_transform(cluster_centers)
        ax.scatter(original_centers[:, 0], original_centers[:, 1], c='red', marker='X', s=200, label='Centroids')
        plt.title("Cluster Centroids")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        cluster_centroids_plot = _save_plot(fig)

        info('正在绘制各聚类的均值对比图')
        fig, ax = plt.subplots(figsize=(10, 6))
        self.cluster_summary.T.plot(kind='bar', ax=ax)
        plt.title("Cluster Feature Means")
        plt.ylabel("Mean Value")
        plt.xlabel("Features")
        plt.legend(title="Cluster")
        cluster_means_plot = _save_plot(fig)

        self.cluster_distribution_plot = cluster_distribution_plot
        self.cluster_centroids_plot = cluster_centroids_plot
        self.cluster_means_plot = cluster_means_plot
