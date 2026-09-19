from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch
from utils.common_utils import default_logger


#定位到backend目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # DeepSeek uses an OpenAI-compatible API. Leave the key empty to use GGUF only.
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    DATA_DIR: str = f"{BASE_DIR}/data"
    MODEL_DIR:str = f"{DATA_DIR}/models"
    #微调后的模型目录
    MODELS_PTH_DIR: str = f"{MODEL_DIR}/pth"
    #原始模型目录
    MODEL_GGUF_DIR: str = f"{MODEL_DIR}/gguf"
    MODEL_SAFETENSORS_DIR: str = f"{MODEL_DIR}/safetensors"
    #图像识别模型目录
    RESNET50_MODEL_ID: str = "microsoft/resnet-50"
    RESNET50_MODEL_PATH: str = \
        f"{MODEL_SAFETENSORS_DIR}/{RESNET50_MODEL_ID}"
    #微调后的图像识别模型目录
    RESNET50_FINETUNED_PTH_DIR: str = \
        f"{MODELS_PTH_DIR}/resnet-50-finetuned"
    RESNET50_FINETUNED_PTH_PATH: str = \
            f"{RESNET50_FINETUNED_PTH_DIR}/resnet-50-finetuned.PTH"
    #微调后的模型所能够识别的类型列表
    JSON_DIR: str = f"{DATA_DIR}/json"
    INSPECTION_RECORD_FILE: str = f"{JSON_DIR}/inspection_records.json"
    KNOWLEDGE_CATALOG_FILE: str = f"{JSON_DIR}/knowledge_catalog.json"
    KEYWORD_KNOWLEDGE_FILE: str = f"{JSON_DIR}/keyword_knowledge.json"
    RESNET50_FINETUNED_JSON_DIR: str = f"{JSON_DIR}/resnet_50_finetuned"
    RESNET50_FINETUNED_CLASSNAMES_PATH: str = \
        f"{RESNET50_FINETUNED_JSON_DIR}/class_names.json"
        
        
    #数据集
    UPLOAD_DATASET_UNZIPED_DIR: str = f"{DATA_DIR}/upload/dataset/unzipped"
    #显存检测
    DEVICE: str = "cpu"
    LLM_GPU_LAYERS: int = 0 #默认0层，不使用CPU
    GPU_MEMORY_THRESHOLD_GB: float = 5.9#默认6GB，超过6GB时使用GPU
    #微调参数
    BATCH_SIZE: int = 16 #默认16
    CONFIDENCE_THRESHOLD: float = 0.25 #默认0.25
    FULL_EPOCHS: int = 10 #默认10（实用中最少40）
    # 空闲
    TRAIN_STATUS_IDLE:str ="idle"
    #运行
    TRAIN_STATUS_RUNNING: str ="running"
    #失败
    TRAIN_STATUS_FAILED:str = "failed"
    #完成
    TRAIN_STATUS_DONE:str = "done"
    #RAG相关
    CHROMA_PERSIST_DIR:str = f"{DATA_DIR}/chroma_db"
    #向量化模型目录
    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_PATH: str = \
        f"{MODEL_SAFETENSORS_DIR}/{EMBEDDING_MODEL_ID}"
    
    
    K:int = 4#默认所搜4个文档相关
    #允许扩展名
    ALLOWED_EXTENSIONS: list = [".pdf",".docx",".doc",".txt"]
    UPLOAD_FILE_DIR:str = f"{DATA_DIR}/upload/files"
    MAX_UPLOAD_FILE_SIZE:int = 1024 * 1024 * 10
    #LLM模型目录
    LLM_MODEL_ID: str = "LoveSeaW/Qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    LLM_MODEL_PATH: str = \
        f"{MODEL_GGUF_DIR}/{LLM_MODEL_ID}"
        # 上下文大小(一次性最多处理的token数)
    LLM_CONTEXT_SIZE: int = 2048
    # 线程数
    LLM_THREADS:int = 4
    def _detect_and_configure_device(self):
        
        #检测并配置设备
        
        #重置运算设备
        self.DEVICE = "cpu"
        self.LLM_GPU_LAYERS = 0
        if torch.cuda.is_available():
            #检查计算设备是否满足条件
            device_inx = torch.cuda.current_device()
            total_memory_bytes = \
            torch.cuda.get_device_properties(device_inx).total_memory
            #检查显存是否足够
            total_memory_gb = total_memory_bytes / (1024**3)
            if total_memory_gb >= self.GPU_MEMORY_THRESHOLD_GB:
                self.DEVICE = "cuda"
                self.LLM_GPU_LAYERS = -1
                default_logger.info(f"使用GPU,显存:{total_memory_gb:.2f}GB")
            else:
                default_logger.warning(f"显存不足,使用CPU")
        else:
            default_logger.warning("CUDA不可用,使用CPU")
        #保存参数
        torch.set_default_device(self.DEVICE)
        default_logger.info(f"默认设备:{self.DEVICE}")
        
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._detect_and_configure_device()
        
settings = Settings()

for d in [
    settings.MODELS_PTH_DIR
    ,settings.RESNET50_FINETUNED_PTH_DIR
    ,settings.RESNET50_FINETUNED_JSON_DIR
    ,settings.UPLOAD_DATASET_UNZIPED_DIR
    ,settings.CHROMA_PERSIST_DIR
    ,settings.UPLOAD_FILE_DIR
]:
    Path(d).mkdir(parents=True,exist_ok=True)
    
default_logger.info("配置完成")
