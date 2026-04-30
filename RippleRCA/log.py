import logging
import os

from datetime import datetime


def get_cur_time():
    # 加上秒级时间戳，避免同一分钟内多次运行导致日志文件名冲突
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


class Logger():
    base_log_path = '{}.log'.format(get_cur_time())

    def __init__(self, loggername, log_path_prefix=None, loglevel=logging.DEBUG):
        '''
           指定保存日志的文件路径，日志级别，以及调用文件
           将日志存入到指定的文件中
        '''
        if log_path_prefix is None:
            log_path = self.base_log_path
        else:
            log_path = log_path_prefix + self.base_log_path

        log_path = os.path.join(os.path.dirname(__file__), 'log', log_path)
        # 确保日志目录存在，避免某些机器/并发情况下 FileHandler 打开失败
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # 创建一个logger
        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(loglevel)
        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(log_path)
        fh.setLevel(loglevel)
        if not self.logger.handlers:
            # 再创建一个handler，用于输出到控制台
            ch = logging.StreamHandler()
            ch.setLevel(loglevel)
            formatter = logging.Formatter('[%(levelname)s]%(asctime)s %(filename)s:%(lineno)d: %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
