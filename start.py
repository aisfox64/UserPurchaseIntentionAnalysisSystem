import base64
import sys

from PIL import Image
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from logger import info
from ui import UiForm
from memory_pic import gyk_png
from data_analysis import DataAnalysis
from data_model_training import DataModelTraining
from data_prediction import DataPrediction

if sys.platform == "win32":
    import ctypes

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("WinID")

lock = 0
save_path = ''  # 保存路径
file_path = ''  # 文件路径
modeSwitching = 0  # 用于记录处理模式，0为分析模式，1为训练模式


# 数据分析线程
class AnalysisThread(QThread):
    def __init__(self, init_file_path):
        super().__init__()
        self.file_path = init_file_path
        self.analysis = None

    def run(self):
        try:
            info('开始数据预览与分析！')
            analysis = DataAnalysis(self.file_path)
            analysis.preview_data()
            analysis.clean_data()
            analysis.analyze_correlation()
            analysis.perform_clustering()
            self.analysis = analysis
            info('分析完成！')
        except Exception as e:
            info(f'错误: {str(e)}')


# 模型训练线程
class TrainThread(QThread):
    def __init__(self, init_file_path):
        super().__init__()
        self.file_path = init_file_path
        self.train = None

    def run(self):
        try:
            info('开始模型训练！')
            train = DataModelTraining(self.file_path)
            train.preview_data()
            train.clean_data()
            train.calculate_vif()
            n_components = train.perform_pca()
            train.train_linear_regression(n_components)
            train.train_random_forest()
            self.train = train
            info('模型训练完成！')
        except Exception as e:
            info(f'错误: {str(e)}')


# 模型训练线程
class DataPredictThread(QThread):
    def __init__(self, init_file_path):
        super().__init__()
        self.file_path = init_file_path
        self.predictor = None
        self.linear_data = None
        self.rf_data = None

    def run(self):
        try:
            info('开始数据预测！')
            predictor = DataPrediction(self.file_path)
            predictor.preview_data()
            predictor.clean_data()
            predictor.load_models()
            x_test_pca_reduced = predictor.preprocess_data()

            # 线性回归预测
            lr_predictions = predictor.predict_with_linear_regression(x_test_pca_reduced)
            self.linear_data = lr_predictions
            info('线性回归预测结果：')
            print(lr_predictions)

            # 随机森林预测
            rf_predictions = predictor.predict_with_random_forest()
            self.rf_data = rf_predictions
            info('随机森林预测结果：')
            print(rf_predictions)
            predictor.drop_comparison_chart()
            self.predictor = predictor
            info('数据预测完成！')
        except Exception as e:
            info(f'错误: {str(e)}')


# ------------------------捕获输出内容------------------------#

class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)

    # 信号发射
    def write(self, text):
        self.newText.emit(str(text))

    def flush(self):
        pass


# ------------------------窗体初始化类------------------------#

class MainWindow(QWidget, UiForm):
    def __init__(self):
        super().__init__()
        self.prediction_thread = None
        self.train_thread = None
        self.analysis_thread = None
        self.mainWindow_UI()  # 窗体结构UI
        self.init_EventFilter()  # 初始化事件过滤器
        self.init_MainWindow()  # 初始化窗体设置
        self.all_Qss()  # QSS样式表
        self.move_drag = False
        self.button10.hide()

        # ------------------------槽函数连接------------------------#

        sys.stdout = Stream(newText=self.onUpdateText)
        self.button3.clicked.connect(self.selectTheFile)
        self.button4.clicked.connect(self.start_btn_1)
        self.button6.clicked.connect(self.start_btn_2)
        self.button10.clicked.connect(self.SingleFile)
        self.button11.clicked.connect(self.batchProcessing)

        self.button1001.clicked.connect(self.show_correlation_plot)
        self.button1002.clicked.connect(self.show_clustering_diagnostics_plot)
        self.button1003.clicked.connect(self.show_cluster_distribution_plot)
        self.button1004.clicked.connect(self.show_cluster_centroids_plot)
        self.button1005.clicked.connect(self.show_cluster_means_plot)

        self.button2001.clicked.connect(self.show_comparison_chart)

    # ------------------------功能函数------------------------#
    def SingleFile(self):
        global modeSwitching, file_path
        modeSwitching = 0
        file_path = ''
        self.button3.setText('请选择测试集')
        self.button4.setText('开始预测')
        self.button6.setText('导出SCV')
        self.button10.hide()
        self.button11.show()
        self.button1001.hide()
        self.button1002.hide()
        self.button1003.hide()
        self.button1004.hide()
        self.button1005.hide()
        self.button2001.show()

    def batchProcessing(self):
        global modeSwitching, file_path
        modeSwitching = 1
        file_path = ''
        self.button3.setText('请选择训练集')
        self.button4.setText('分析数据')
        self.button6.setText('训练模型')
        self.button11.hide()
        self.button10.show()
        self.button1001.show()
        self.button1002.show()
        self.button1003.show()
        self.button1004.show()
        self.button1005.show()
        self.button2001.hide()

    # 更新窗口信息
    def onUpdateText(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.insertText(text)

    # 文件选择
    def selectTheFile(self):
        global file_path
        filePath, filetype = QFileDialog.getOpenFileName(self, "选择文件", "./", "CSV Files (*.csv)")
        if filePath:
            file_path = filePath
            self.textBrowser.clear()
            info(f'已选中的文件：{filePath}\n')
            self.button3.setText('（已选择文件）\n点击重新选择')
        else:
            pass

    # 开始入口
    def start_btn_1(self):
        if file_path:
            if modeSwitching == 0:
                self.prediction_thread = DataPredictThread(file_path)
                self.prediction_thread.start()
            else:
                self.analysis_thread = AnalysisThread(file_path)
                self.analysis_thread.start()
        else:
            QMessageBox.warning(self, "提示", "请选择文件", QMessageBox.Cancel)

    def start_btn_2(self):
        if file_path:
            if modeSwitching == 0:
                self.prediction_thread.predictor.save_predictions(self.prediction_thread.linear_data, "./out/linear_regression_predictions.csv")
                self.prediction_thread.predictor.save_predictions(self.prediction_thread.rf_data, "./out/random_forest_predictions.csv")
                info(f'已保存文件到根out目录下')
            else:
                self.train_thread = TrainThread(file_path)
                self.train_thread.start()
        else:
            QMessageBox.warning(self, "提示", "请选择文件", QMessageBox.Cancel)

    def show_correlation_plot(self):
        try:
            image = Image.open(self.analysis_thread.analysis.correlation_plot)
            image.show()
        except:
            QMessageBox.warning(self, "提示", "请先分析数据", QMessageBox.Cancel)

    def show_clustering_diagnostics_plot(self):
        try:
            image = Image.open(self.analysis_thread.analysis.clustering_diagnostics_plot)
            image.show()
        except:
            QMessageBox.warning(self, "提示", "请先分析数据", QMessageBox.Cancel)

    def show_cluster_distribution_plot(self):
        try:
            image = Image.open(self.analysis_thread.analysis.cluster_distribution_plot)
            image.show()
        except:
            QMessageBox.warning(self, "提示", "请先分析数据", QMessageBox.Cancel)

    def show_cluster_centroids_plot(self):
        try:
            image = Image.open(self.analysis_thread.analysis.cluster_centroids_plot)
            image.show()
        except:
            QMessageBox.warning(self, "提示", "请先分析数据", QMessageBox.Cancel)

    def show_cluster_means_plot(self):
        try:
            image = Image.open(self.analysis_thread.analysis.cluster_means_plot)
            image.show()
        except:
            QMessageBox.warning(self, "提示", "请先分析数据", QMessageBox.Cancel)

    def show_comparison_chart(self):
        try:
            image = Image.open(self.prediction_thread.predictor.comparison_chart)
            image.show()
        except:
            QMessageBox.warning(self, "提示", "请先预测数据", QMessageBox.Cancel)

    # 导出excel
    def save_to_csv(self):
        global save_path, lock
        try:
            if lock == 0:
                pass
            else:
                QMessageBox.warning(self, "提示", "请等待计算结束", QMessageBox.Cancel)
        except:
            QMessageBox.warning(self, "提示", "数据有误", QMessageBox.Cancel)

    # ---------------------------窗体相关设置--------------------------#

    def init_EventFilter(self):  # 初始化事件过滤器
        self.widget.installEventFilter(self)
        self.widget2.installEventFilter(self)

    def init_MainWindow(self):  # 设置窗体样式
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

    # 最小化
    @pyqtSlot()
    def on_button1_clicked(self):
        self.showMinimized()

    # 退出
    @pyqtSlot()
    def on_button2_clicked(self):
        self.close()

    # 重写窗口拖动的3个方法
    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton) and (event.y() < self.widget.height()):
            self.move_drag = True
            self.move_DragPosition = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.move_drag:
            self.move(QMouseEvent.globalPos() - self.move_DragPosition)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.move_drag = False


# ---------------------------执行主体--------------------------#

if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = MainWindow()
    Logo = QPixmap()
    Logo.loadFromData(base64.b64decode(gyk_png))
    icon = QIcon()
    icon.addPixmap(Logo, QIcon.Normal, QIcon.Off)
    myWin.move(50, 50)
    myWin.setWindowIcon(icon)
    myWin.show()
    sys.exit(app.exec_())
