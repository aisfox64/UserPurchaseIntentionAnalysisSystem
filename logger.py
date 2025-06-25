import datetime

# ANSI 转义序列用于设置文本颜色
GREEN = '\033[92m'
RESET = '\033[0m'


def green_print(message):
    """
    以绿色文本打印消息到控制台。

    :param message: 要打印的消息
    """
    print(f"{GREEN}{message}{RESET}")


def info(message):
    """
    模仿 logging.info() 的行为，以绿色文本打印带有时间戳的日志信息。

    :param message: 要记录的日志信息
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - INFO - {message}"
    print(log_message)
