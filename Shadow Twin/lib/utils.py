import logging
from torch import Tensor
import torch.nn.functional as f

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    l.setLevel(level)

    # 清除旧的 handlers，防止句柄泄露
    if l.hasHandlers():
        for handler in l.handlers:
            handler.close()
        l.handlers.clear()

    formatter = logging.Formatter('%(asctime)s : %(message)s')

    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)

    return l