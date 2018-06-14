import logging
import os
from datetime import datetime


def work_pre(output_folder):
    logger = logging.getLogger("CTR")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        print(logger.handlers)
    else:
        path = os.getcwd()
        start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        output_folder = path + "/../Output/" + start_time + "/"
        # relativeOutputFolder = "/../Output/" + start_time + "/"
        os.mkdir(output_folder)
        filename = output_folder + "OperationRecord.log"
        # 建立一个file handler来把日志记录在文件里，级别为debug以上
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        # 建立一个stream handler来把日志打在CMD窗口上，级别为error以上
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # 设置日志格式
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # 将相应的handler添加在logger对象中
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger, output_folder
