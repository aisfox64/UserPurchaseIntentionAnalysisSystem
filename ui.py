import base64

from PyQt5 import QtCore, QtGui, QtWidgets

from memory_pic import gyk_png, wjj_png

logo_1 = base64.b64decode(gyk_png)
logo_2 = base64.b64decode(wjj_png)


class UiForm(object):
    def mainWindow_UI(self):
        # 主窗口
        self.setObjectName("MianWindow")
        self.resize(1280, 680)
        # 顶部控件可移动区域
        self.widget = QtWidgets.QWidget(self)
        self.widget.setObjectName('widget')
        self.widget.resize(1280, 50)
        # 主内容区域
        self.widget2 = QtWidgets.QWidget(self)
        self.widget2.setObjectName('widget2')
        self.widget2.resize(1280, 630)
        self.widget2.move(0, 50)
        # 初始化base64图标
        self.pixmap_1 = QtGui.QPixmap()
        self.pixmap_1.loadFromData(logo_1)
        self.pixmap_2 = QtGui.QPixmap()
        self.pixmap_2.loadFromData(logo_2)
        # 最小化按钮
        self.button1 = QtWidgets.QPushButton(self.widget)
        self.button1.setObjectName('button1')
        self.button1.setText('0')
        self.button1.resize(30, 30)
        self.button1.move(1200, 10)
        # 关闭按钮
        self.button2 = QtWidgets.QPushButton(self.widget)
        self.button2.setObjectName('button2')
        self.button2.setText('r')
        self.button2.resize(30, 30)
        self.button2.move(1240, 10)
        # 左部布局框架
        self.backgroundlabel_ltp = QtWidgets.QLabel(self.widget2)
        self.backgroundlabel_ltp.setObjectName('backgroundlabel_ltp')
        self.backgroundlabel_ltp.resize(260, 630)
        self.backgroundlabel_ltp.move(0, 0)
        # 右部布局框架
        self.backgroundlabel_rtp = QtWidgets.QLabel(self.widget2)
        self.backgroundlabel_rtp.setObjectName('backgroundlabel_rtp')
        self.backgroundlabel_rtp.resize(200, 630)
        self.backgroundlabel_rtp.move(1080, 0)
        # 模式切换按钮背景
        self.backgroundlabe2 = QtWidgets.QLabel(self.widget2)
        self.backgroundlabe2.setObjectName('backgroundlabe2')
        self.backgroundlabe2.resize(100, 26)
        self.backgroundlabe2.move(80, 10)
        # 模式切换按钮背景
        self.backgroundlabe3 = QtWidgets.QLabel(self.widget2)
        self.backgroundlabe3.setObjectName('backgroundlabe3')
        self.backgroundlabe3.resize(100, 26)
        self.backgroundlabe3.move(80, 10)
        # 应用标题
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName('label')
        self.label.setText('用户购票意愿分析系统')
        self.label.resize(200, 40)
        self.label.move(60, 5)
        # 应用图标
        self.label2 = QtWidgets.QLabel(self.widget)
        self.label2.setObjectName('label2')
        self.label2.setPixmap(self.pixmap_1)
        self.label2.resize(40, 40)
        self.label2.move(10, 5)
        # 选项按钮
        self.label6 = QtWidgets.QLabel(self.widget2)
        self.label6.setObjectName('label6')
        self.label6.setText('系统日志信息:')
        self.label6.resize(200, 30)
        self.label6.move(265, 10)
        self.label7 = QtWidgets.QLabel(self.widget2)
        self.label7.setObjectName('label7')
        self.label7.setPixmap(self.pixmap_2)
        self.label7.resize(100, 80)
        self.label7.move(80, 110)
        self.button3 = QtWidgets.QPushButton(self.widget2)
        self.button3.setObjectName('button3')
        self.button3.setText('请选择测试集')
        self.button3.resize(140, 100)
        self.button3.move(60, 100)
        self.button4 = QtWidgets.QPushButton(self.widget2)
        self.button4.setObjectName('button4')
        self.button4.setText('开始分析')
        self.button4.resize(180, 40)
        self.button4.move(40, 500)
        self.button6 = QtWidgets.QPushButton(self.widget2)
        self.button6.setObjectName('button6')
        self.button6.setText('导出SCV')
        self.button6.resize(180, 40)
        self.button6.move(40, 560)
        self.button10 = QtWidgets.QPushButton(self.widget2)
        self.button10.setObjectName('button10')
        self.button10.setText('当前为模型训练模式')
        self.button10.resize(180, 26)
        self.button10.move(40, 10)
        self.button11 = QtWidgets.QPushButton(self.widget2)
        self.button11.setObjectName('button11')
        self.button11.setText('当前为数据分析模式')
        self.button11.resize(180, 26)
        self.button11.move(40, 10)
        # 输出信息文本框
        self.textBrowser = QtWidgets.QTextBrowser(self.widget2)
        self.textBrowser.setObjectName('textBrowser')
        self.textBrowser.resize(820, 520)
        self.textBrowser.move(260, 40)

        self.button1001 = QtWidgets.QPushButton(self.widget2)
        self.button1001.setObjectName('button4')
        self.button1001.setText('显示相关性热力图')
        self.button1001.resize(160, 30)
        self.button1001.move(1100, 50)
        self.button1001.hide()
        self.button1002 = QtWidgets.QPushButton(self.widget2)
        self.button1002.setObjectName('button4')
        self.button1002.setText('显示聚类诊断图')
        self.button1002.resize(160, 30)
        self.button1002.move(1100, 100)
        self.button1002.hide()
        self.button1003 = QtWidgets.QPushButton(self.widget2)
        self.button1003.setObjectName('button4')
        self.button1003.setText('显示聚类结果散点图')
        self.button1003.resize(160, 30)
        self.button1003.move(1100, 150)
        self.button1003.hide()
        self.button1004 = QtWidgets.QPushButton(self.widget2)
        self.button1004.setObjectName('button4')
        self.button1004.setText('显示聚类中心图')
        self.button1004.resize(160, 30)
        self.button1004.move(1100, 200)
        self.button1004.hide()
        self.button1005 = QtWidgets.QPushButton(self.widget2)
        self.button1005.setObjectName('button4')
        self.button1005.setText('各聚类的均值对比图')
        self.button1005.resize(160, 30)
        self.button1005.move(1100, 250)
        self.button1005.hide()

        self.button2001 = QtWidgets.QPushButton(self.widget2)
        self.button2001.setObjectName('button4')
        self.button2001.setText('显示预测结果对比图')
        self.button2001.resize(160, 30)
        self.button2001.move(1100, 50)

        QtCore.QMetaObject.connectSlotsByName(self)

    # -------------------------QSS样式表-------------------------#

    def all_Qss(self):
        qssStyle = '''
                    QWidget#widget{
                    background-color:#b2b2b2;
                    border-top-left-radius: 18px;
                    border-top-right-radius: 18px;}

                    QWidget#widget2{
                    background-color:#f1f1f1;
                    border-bottom-left-radius: 18px;
                    border-bottom-right-radius: 18px;}

                    QPushButton#button1{
                    font-family:"Webdings";
                    background:#6DDF6D;
                    border-radius:5px;
                    border:none;
                    font-size:18px;
                    }
                    QPushButton#button1:hover{
                    background:green;
                    }
                   
                    QPushButton#button2{
                    font-family:"Webdings";
                    background:#F76677;
                    border-radius:5px;
                    border:none;
                    font-size:18px;
                    }
                    QPushButton#button2:hover{
                    background:red;
                    }

                    QPushButton#button3{
                    color:#333333;
                    font-size:16px;
                    background:transparent;
                    border: 2px solid #333333;
                    border-radius:16px;
                    }
                    QPushButton#button3:hover{
                    color:#333333;
                    font-size:16px;
                    background-color: rgba(143,143,143,0.5);
                    border: 2px solid #333333;
                    border-radius:16px;
                    }

                    QPushButton#button4,#button5{
                    font: 16px "微软雅黑";
                    font-weight:bold;
                    border-radius:6px;
                    border:1px solid #252525;
                    background-color: #c5c5c5;
                    }
                    QPushButton#button4:hover{
                    font: 18px "微软雅黑";
                    font-weight:bold;
                    border-radius:6px;
                    border:2px solid #252525;
                    background-color: #b2b2b2;
                    }
                    QPushButton#button5:hover{
                    font: 18px "微软雅黑";
                    font-weight:bold;
                    border-radius:6px;
                    border:2px solid #252525;
                    background-color: #b2b2b2;
                    }

                    QPushButton#button6,#button7{
                    font: 16px "微软雅黑";
                    font-weight:bold;
                    border-radius:6px;
                    border:1px solid #252525;
                    background-color: #c5c5c5;
                    }
                    QPushButton#button6:hover{
                    font: 18px "微软雅黑";
                    font-weight:bold;
                    border-radius:6px;
                    border:2px solid #252525;
                    background-color: #b2b2b2;
                    }
                    QPushButton#button7:hover{
                    font: 18px "微软雅黑";
                    font-weight:bold;
                    border-radius:6px;
                    border:2px solid #252525;
                    background-color: #b2b2b2;
                    }

                    QPushButton#button8{
                    font: 16px "微软雅黑";
                    background:#c5c5c5;
                    border:1px solid #252525;
                    border-top-left-radius:8px;
                    border-top-right-radius:8px;
                    }
                    QPushButton#button8:hover{
                    font: 18px "微软雅黑";
                    background:#b2b2b2;
                    border:2px solid #252525;
                    border-top-left-radius:8px;
                    border-top-right-radius:8px;
                    }

                    QPushButton#button9{
                    font: 16px "微软雅黑";
                    background:#c5c5c5;
                    border:1px solid #252525;
                    border-bottom-left-radius:8px;
                    border-bottom-right-radius:8px;
                    }
                    QPushButton#button9:hover{
                    font: 18px "微软雅黑";
                    background:#b2b2b2;
                    border:2px solid #252525;
                    border-bottom-left-radius:8px;
                    border-bottom-right-radius:8px;
                    }
                    
                    QPushButton#button10{
                    font: 12px "微软雅黑";
                    font-weight:bold;
                    border-radius:12px;
                    border:1px solid #252525;
                    background-color: #c5c5c5;
                    }
                    QPushButton#button10:hover{
                    font: 12px "微软雅黑";
                    font-weight:bold;
                    border-radius:12px;
                    border:2px solid #252525;
                    background-color: #b2b2b2;
                    }

                    QPushButton#button11{
                    font: 12px "微软雅黑";
                    font-weight:bold;
                    border-radius:12px;
                    border:1px solid #252525;
                    background-color: #c5c5c5;
                    }
                    QPushButton#button11:hover{
                    font: 12px "微软雅黑";
                    font-weight:bold;
                    border-radius:12px;
                    border:2px solid #252525;
                    background-color: #b2b2b2;
                    }

                    QLabel#label{
                    font-family:"微软雅黑";
                    color:#060606;
                    font-size:20px;
                    }

                    QLabel#label3,#label4,#label6,#label5{
                    font-family:"微软雅黑";
                    font-weight:bold;
                    color:#252525;
                    font-size:20px;
                    }

                    QLabel#backgroundlabel_ltp{
                    background:#e1e1e1;
                    border-right:1px solid #575757;
                    border-bottom-left-radius: 18px;
                    }
                    
                    QLabel#backgroundlabel_rtp{
                    background:#e1e1e1;
                    border-left:1px solid #575757;
                    border-bottom-right-radius: 18px;
                    }

                    QLabel#backgroundlabe2{
                    font-family:"微软雅黑";
                    font-weight:bold;
                    border-radius:12px;
                    border:2px solid #252525;
                    background-color: #7b7b7b;
                    }

                    QTextBrowser#textBrowser{
                    font: 16px "微软雅黑";
                    background:#f1f1f1;
                    border-top:1px solid #575757;
                    border-bottom:1px solid #7b7b7b;
                    }

                   '''
        self.setStyleSheet(qssStyle)
