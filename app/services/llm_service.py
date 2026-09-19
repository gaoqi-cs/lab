import threading
import torch
from openai import OpenAI
from config.config import settings
from llama_cpp import Llama
from utils.common_utils import default_logger
from typing import List,Dict,Optional


class LLMService:
    #单例模式
    _instance = None
    _lock = threading.Lock()
    def __new__(cls):
        
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self,'_initialized'):
            return
        self._initialized = True
        
        self.model_path = settings.LLM_MODEL_PATH
        self.llm = None
        self._loaded = False
        self.deepseek_client = None
        if settings.DEEPSEEK_API_KEY:
            self.deepseek_client = OpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL
            )
            default_logger.info(
                f"DeepSeek 云端问答已启用（模型：{settings.DEEPSEEK_MODEL}）"
            )

    def _load_model(self):
        if self._loaded:
            return
        n_gpu_layers = settings.LLM_GPU_LAYERS
        try:
            #加载压缩后的模型
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=settings.LLM_CONTEXT_SIZE,
                n_gpu_layers=n_gpu_layers,
                n_threads=settings.LLM_THREADS,
                #是否打印详细信息
                verbose=False
                
            )
            #标记模型已加载
            self._loaded = True
            default_logger.info(f"模型加载完成(设备:{settings.DEVICE})")
        except Exception as e:
            default_logger.error(f"模型加载失败:{e}")
            self._loaded = False
            self.llm = None

    def is_ready(self)->bool:
        #检查模型是否已加载
        return self._loaded and self.llm is not None

    def _chat_with_deepseek(self, messages, temperature, max_tokens, top_p, stream):
        try:
            # v4-flash can spend a large hidden reasoning budget before producing content.
            provider_max_tokens = max(max_tokens, 4096) if settings.DEEPSEEK_MODEL == "deepseek-v4-flash" else max_tokens
            response = self.deepseek_client.chat.completions.create(
                model=settings.DEEPSEEK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=provider_max_tokens,
                top_p=top_p,
                stream=stream
            )
            if stream:
                def stream_generator():
                    try:
                        for chunk in response:
                            content = chunk.choices[0].delta.content if chunk.choices else None
                            if content:
                                yield content
                    except Exception as e:
                        default_logger.error(f"DeepSeek 流式调用失败：{e}")
                        yield f"错误：{str(e)}"
                return stream_generator()
            content = response.choices[0].message.content if response.choices else None
            return content or "DeepSeek 未返回内容，请重试"
        except Exception as e:
            default_logger.error(f"DeepSeek 调用失败：{e}")
            return f"DeepSeek 调用失败：{str(e)}，请检查密钥和网络后重试"
    
    def chat(self,messages:List[Dict[str,str]],temperature:float=0.7, max_tokens:int=512,top_p:float=0.9,stream:bool=False):
        if self.deepseek_client is not None:
            return self._chat_with_deepseek(messages, temperature, max_tokens, top_p, stream)
        if not self.is_ready():
            default_logger.error("模型未加载完成,重新载入")
            self._load_model()
        if not self.is_ready():
            #【修改1】修正错别字："为加载完成" → "未加载完成"
            error_msg = "模型未加载完成,请检查模型路径是否正确"
            if stream:
                def error_gen():
                    #发送错误信息(异步流式返回)
                    yield f"data:{error_msg}"
                return error_gen
            #【修改2】新增return！模型加载失败直接返回错误文本，不要继续往下执行try
            return error_msg
        
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                #每一次提问或者回答的结束标记,整个会话的结束标记
                #防止模型在回答完问题后,继续自言自语的问题
                stop=["<|im_end|>","<|endoftext|>"]
            )
            
            if stream:
                def stream_generator():
                    #异步流式返回
                    try:
                        #遍历模型
                        for chunk in response:
                            if 'choices' in chunk and len(chunk['choices'])>0:
                                #【修改3】修复原有嵌套delta的bug！原代码：delta = chunk['choices'][0]['delta'].get('delta',{})
                                #返回结果第一层就是delta字典，不要再额外get('delta')
                                delta = chunk['choices'][0]['delta']
                                content= delta.get('content','')
                                if content:
                                    #发送内容（异步流式返回）
                                    yield content
                    except Exception as e:
                        default_logger.error(f"异步流式返回失败:{e}")
                        yield f"错误:{str(e)}"
                return stream_generator()
            else:
                if 'choices' in response and len(response['choices'])>0:
                    #【修改4】增加判空，防止content为None，上层generate把None转为空字符串""，页面只剩备注文字
                    content = response['choices'][0]['message'].get('content')
                    if content is not None:
                        return content
                    else:
                        default_logger.error("模型返回内容为None")
                        return "模型未输出内容,请重试"
                else:
                    default_logger.error("模型返回为空")
                    return "模型返回为空,请重试"
        except Exception as e:
            default_logger.error(f"LLM模型调用失败:{e}")
            return f"模型调用失败:{str(e)},请重试"

    def generate(self,user_prompt:str,system_prompt:Optional[str]=None,temperature:float=0.7, max_tokens:int=512)->str:
        messages = []
        #系统提示
        if system_prompt:
            messages.append({"role":"system","content":system_prompt})
        #用户提示（含用户问题）
        messages.append({"role":"user","content":user_prompt})
        result = self.chat(messages,temperature,max_tokens)
        #处理模型返回为空的情况
        return result if isinstance(result,str) else ""
    

llm_service = LLMService()
