from pathlib import Path

#定位到backend目录
BASE_DIR=Path(__file__).resolve().parent.parent.parent
#日志的位置
_LOG_FILE = f"{BASE_DIR}/data/logs/app.log"
_LOGGER_CACHE = {}
import logging
import sys
from logging.handlers import RotatingFileHandler
 

def setup_logger(
    name: str="校园实验室安全巡检助手",
    log_file: str =None,
    level: str ="INFO",
    max_bytes: int=1024*1024*10,
    backup_count: int=5
    
)->logging.Logger:
    #获得日志对象
    logger =logging.getLogger(name)
    logger.setLevel(getattr(logging,level.upper()))
    if logger.handlers:
        #如果logger已经存在 直接返回
        return logger
    #控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(   
    "%(asctime)s - %(name)s-%(levelname)s-%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    #将控制台处理器放到logger中
    logger.addHandler(console_handler)
    #文件输出相关 
    if log_file:
        try:
            #将字符串封装为路径
            log_path=Path(log_file)
            #强制创建该文件的文件夹
            log_path.parent.mkdir(parents=True,exist_ok=True)
            #创建文件处理器
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8"
            )
            
            file_handler.setLevel(logging.INFO)
            file_formatter=logging.Formatter(
                 "%(asctime)s - %(name)s-%(levelname)s-%(message)s",
                  datefmt="%Y-%m-%d %H:%M:%S"
            )
        
            file_handler.setFormatter(file_formatter)
            #将文件处理器放到logger
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"无法创建日志文件:{e}")
    return logger

def get_logger(name:str="校园实验室安全巡检助手")->logging.Logger:
    #从缓存中获取logger（保证每个name只有一个logger）
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    #如果缓存中没有，创建新的logger
    logger =setup_logger(name,log_file=_LOG_FILE,level="INFO")
    #缓存logger
    _LOGGER_CACHE[name]=logger
    return logger

#默认日志对象
default_logger =get_logger()
