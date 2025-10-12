import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # 加载.env文件中的环境变量


@dataclass
class Config:
    """基础配置类"""
    BASE_MODEL: str = os.getenv("BASE_MODEL")
    MODEL_VERSION:str = os.getenv("MODEL_VERSION")
    TRAIN_MODEL:str = f"{BASE_MODEL}/{MODEL_VERSION}"
    OUTPUT_DIR:str = f"output/{MODEL_VERSION}"
    DATA_USE_FRAC:float = float(os.getenv("DATA_USE_FRAC", default=0.002))



# 创建配置实例
global_config = Config()