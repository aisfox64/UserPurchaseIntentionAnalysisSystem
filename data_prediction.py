from io import BytesIO

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from logger import info


def _save_plot(fig):
    """将 matplotlib 图保存到内存中"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    return buffer


class DataPrediction:
    def __init__(self, test_file_path, model_dir='model'):
        self.test_data = pd.read_csv(test_file_path)
        self.model_dir = model_dir
        self.scaler = None
        self.pca = None
        self.linear_model = None
        self.rf_model = None

        self.linear_data = None
        self.rf_data = None

        self.comparison_chart = None

    def preview_data(self):
        info('####################数据预览####################')
        info(f'测试集纬度：{self.test_data.shape}')
        info("数据信息：")
        print(self.test_data.info())
        info(f'缺失值情况：\n{self.test_data.isna().sum()}')

    def clean_data(self):
        info('####################数据清洗####################')

        # 删除ID列
        if 'UserID' in self.test_data.columns:
            info('删除ID列')
            self.test_data.drop(['UserID'], axis=1, inplace=True)

        # 处理缺失值
        info('处理缺失值')
        for column in self.test_data.columns:
            if self.test_data[column].isna().sum() > 0:
                if self.test_data[column].dtype in ['float64', 'int64']:
                    median_value = self.test_data[column].median()
                    info(f'数值列 {column} 缺失值将被填充为中位数: {median_value}')
                    self.test_data[column].fillna(median_value, inplace=True)
                else:
                    mode_value = self.test_data[column].mode()[0]
                    info(f'分类列 {column} 缺失值将被填充为众数: {mode_value}')
                    self.test_data[column].fillna(mode_value, inplace=True)

        # 统一处理 Yes/No 字段
        yes_no_columns = [col for col in self.test_data.columns if self.test_data[col].dropna().isin(['Yes', 'No']).all()]
        if yes_no_columns:
            info(f'检测到以下 Yes/No 列：{yes_no_columns}')
            for column in yes_no_columns:
                self.test_data[column] = self.test_data[column].map({'Yes': 1, 'No': 0})

        # 编码分类变量
        info('编码分类变量')
        categorical_columns = self.test_data.select_dtypes(include=['object']).columns
        if categorical_columns.any():
            info(f'检测到以下分类列：{list(categorical_columns)}')
            label_encoder = LabelEncoder()
            for column in categorical_columns:
                self.test_data[column] = label_encoder.fit_transform(self.test_data[column])
                info(f'分类列 {column} 已转换为数字编码')

        # 删除重复值
        duplicate_count = self.test_data.duplicated().sum()
        if duplicate_count > 0:
            info(f'检测到 {duplicate_count} 条重复记录，将被删除')
            self.test_data.drop_duplicates(inplace=True)

        # 标记和处理异常值
        info('检测并处理异常值')
        numeric_columns = self.test_data.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:

            if column == 'Taken_product':
                continue

            Q1 = self.test_data[column].quantile(0.25)
            Q3 = self.test_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((self.test_data[column] < lower_bound) | (self.test_data[column] > upper_bound)).sum()
            if outliers > 0:
                info(f'数值列 {column} 检测到 {outliers} 个异常值，将使用上下限进行裁剪')
                self.test_data[column] = self.test_data[column].clip(lower=lower_bound, upper=upper_bound)

        print(self.test_data.head())
        info('数据清洗完成！')

    def load_models(self):
        info('加载保存的模型')
        self.scaler = joblib.load(f'{self.model_dir}/linear_regression_scaler.pkl')
        self.pca = joblib.load(f'{self.model_dir}/linear_regression_pca.pkl')
        self.linear_model = joblib.load(f'{self.model_dir}/linear_regression_model.pkl')
        self.rf_model = joblib.load(f'{self.model_dir}/random_forest_model.pkl')

    def preprocess_data(self):
        info('对测试数据进行预处理')
        # 数据标准化
        x_test_scaled = self.scaler.transform(self.test_data)
        # 主成分分析
        x_test_pca = self.pca.transform(x_test_scaled)
        # 使用与训练时相同数量的主成分
        n_components = self.pca.n_components
        x_test_pca_reduced = x_test_pca[:, :n_components]
        return x_test_pca_reduced

    def predict_with_linear_regression(self, x_test_pca_reduced):
        info('使用线性回归模型进行预测')
        test_predictions = self.linear_model.predict(x_test_pca_reduced)
        test_data_with_predictions = self.test_data.copy()
        test_data_with_predictions['predicted_taken_product'] = test_predictions
        # 对预测值进行偏移修正
        min_value = test_data_with_predictions['predicted_taken_product'].min()
        offset = abs(min_value) if min_value < 0 else 0
        test_data_with_predictions['predicted_taken_product'] += offset
        self.linear_data = test_data_with_predictions['predicted_taken_product']
        return test_data_with_predictions

    def predict_with_random_forest(self):
        info('使用随机森林模型进行预测')
        test_predictions = self.rf_model.predict(self.test_data)
        test_data_with_predictions = self.test_data.copy()
        test_data_with_predictions['predicted_taken_product'] = test_predictions
        self.rf_data = test_data_with_predictions['predicted_taken_product']
        return test_data_with_predictions

    def drop_comparison_chart(self):
        info('正在绘制对比图')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.linear_data, label='Linear Regression Trend', color='blue', alpha=0.7, marker='o')
        ax.plot(self.rf_data, label='Random Forest Trend', color='green', alpha=0.7, marker='x')
        ax.set_title('Comparison of Predicted Trends', fontsize=16)
        ax.set_xlabel('Data Point Index', fontsize=12)
        ax.set_ylabel('Predicted Value', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        self.comparison_chart = _save_plot(fig)

    def save_predictions(self, predictions, filename):
        predictions.to_csv(filename, index=False)
        info(f'预测结果已保存至 {filename}')
