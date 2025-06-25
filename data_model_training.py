import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from logger import info


class DataModelTraining:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.x = None
        self.y = None
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.models = {}

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

    def calculate_vif(self):
        info('计算特征的VIF值')
        self.x = self.data.drop(['Taken_product'], axis=1)
        self.y = self.data['Taken_product']
        x_with_const = sm.add_constant(self.x)
        vif_data = pd.DataFrame({
            "Feature": x_with_const.columns,
            "VIF": [variance_inflation_factor(x_with_const.values, i) for i in range(x_with_const.shape[1])]
        })
        info(vif_data)
        return vif_data

    def perform_pca(self, variance_threshold=0.95):
        info('执行PCA降维')
        x_scaled = self.scaler.fit_transform(self.x)
        self.pca.fit(x_scaled)
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
        info(f'选取前{n_components}个主成分覆盖累计方差贡献率：{variance_threshold}')
        self.pca.n_components = n_components
        return n_components

    def train_linear_regression(self, n_components):
        info('训练多元线性回归模型')
        x_pca = self.pca.transform(self.scaler.transform(self.x))[:, :n_components]
        x_train, x_test, y_train, y_test = train_test_split(x_pca, self.y, test_size=0.3, random_state=15)
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse, r2 = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)
        info(f"线性回归性能 - MSE: {mse}, R2: {r2}")
        self.models['linear_regression'] = model
        joblib.dump(model, 'model/linear_regression_model.pkl')
        joblib.dump(self.scaler, 'model/linear_regression_scaler.pkl')
        joblib.dump(self.pca, 'model/linear_regression_pca.pkl')
        info("多元线性回归模型已保存！")

    def train_random_forest(self):
        info('训练随机森林模型')
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=10)
        rf = RandomForestRegressor(random_state=15)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        mse, r2 = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)
        info(f"随机森林性能 - MSE: {mse}, R2: {r2}")
        self.models['random_forest'] = rf
        joblib.dump(rf, 'model/random_forest_model.pkl')
        info("随机森林模型已保存！")

    def hyperparameter_tuning(self):
        info('随机森林超参数优化')
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['auto', 'sqrt']
        }
        rf = RandomForestRegressor(random_state=15)
        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=10, cv=5, random_state=17, n_jobs=1
        )
        x_train, _, y_train, _ = train_test_split(self.x, self.y, test_size=0.3, random_state=10)
        random_search.fit(x_train, y_train)
        best_params = random_search.best_params_
        info(f"最佳参数: {best_params}")
        self.models['random_forest_tuned'] = RandomForestRegressor(**best_params, random_state=15)
        joblib.dump(self.models['random_forest_tuned'], 'model/random_forest_tuned_model.pkl')
        info("随机森林调参模型已保存！")

    def visualize_residuals(self, y_test, y_pred):
        info('绘制残差分布图')
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.title("残差分布")
        plt.show()
