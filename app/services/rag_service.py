import json
import re
import threading
from pathlib import Path
from config.config import settings
from utils.common_utils import default_logger
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RagService:
    _instance = None
    _lock = threading.Lock()
    _GENERIC_TERMS = {
        "什么", "怎么", "如何", "是否", "需要", "应该", "可以", "使用",
        "处理", "检查", "安全", "问题", "哪些", "以后", "之前", "以后",
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self._embedding_loaded = False
        self._embedding_model_dir = None
        self.keyword_file = Path(settings.KEYWORD_KNOWLEDGE_FILE)
        self.keyword_file.parent.mkdir(parents=True, exist_ok=True)

        # Chroma 返回的是距离值，距离越小表示内容越相关。
        self.distance_threshold = 0.45

        self._init_chromadb()

    def _init_chromadb(self):
        try:
            self.vectorstore = Chroma(
                collection_name="lab_safety_knowledge",
                embedding_function=None,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
            self.retriever = None
            default_logger.info("向量数据库初始化完成")
        except Exception as e:
            default_logger.error(f"向量数据库初始化失败: {e}")
            self.vectorstore = None

    def _load_embeddings(self):
        if self._embedding_loaded:
            return
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_PATH,
                model_kwargs={"device": 'cpu'},
                encode_kwargs={"normalize_embeddings": True}
            )
            self._embedding_loaded = True
            default_logger.info(" 嵌入模型加载完成")
        except Exception as e:
            default_logger.error(f"嵌入模型加载失败: {e}")
            self.embeddings = None
            self._embedding_loaded = False

    def _ensure_components(self):
        if self.vectorstore is None:
            default_logger.info("向量数据库未初始化,开始初始化")
            return False
        if not self._embedding_loaded:
            self._load_embeddings()
        if not self._embedding_loaded:
            default_logger.error("嵌入模型加载失败")
            return False

        if self.vectorstore._embedding_function is None and self.embeddings is not None:
            self.vectorstore._embedding_function = self.embeddings

        if self.retriever is None and self.embeddings is not None:
            self.retriever = self.vectorstore.as_retriever()

        default_logger.info("所有组件初始化完成")
        return True

    def _load_keyword_chunks(self):
        if not self.keyword_file.exists():
            return []
        try:
            chunks = json.loads(self.keyword_file.read_text(encoding="utf-8"))
            return chunks if isinstance(chunks, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _store_keyword_chunks(self, source, chunks):
        records = [item for item in self._load_keyword_chunks() if item.get("source") != source]
        records.extend({"source": source, "content": chunk.page_content} for chunk in chunks)
        self.keyword_file.write_text(
            json.dumps(records[-5000:], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def _format_local_answer(cls, question, chunks):
        """只摘取最相关段落，避免把整段资料或多个候选答案全部返回。"""
        paragraphs = []
        for chunk in chunks:
            paragraphs.extend(
                paragraph.strip()
                for paragraph in re.split(r"\n\s*\n", chunk)
                if paragraph.strip()
            )

        if not paragraphs:
            return ""

        best_index, best_score = max(
            enumerate(cls._keyword_score(question, paragraph) for paragraph in paragraphs),
            key=lambda item: item[1],
        )
        selected = [paragraphs[best_index]] if best_score else paragraphs[:1]

        # FAQ 的标题和答案通常相邻；标题命中时一并保留下一段答案。
        if (
            best_index + 1 < len(paragraphs)
            and len(selected[0]) < 400
            and selected[0].endswith(("？", "?"))
        ):
            selected.append(paragraphs[best_index + 1])

        content = "\n\n".join(selected)
        return content[:800]

    @classmethod
    def _specific_terms(cls, question):
        """提取可用于判断资料是否真正命中的三到四字中文短语。"""
        phrases = re.findall(r"[\u4e00-\u9fff]{3,}", question)
        terms = set()
        for phrase in phrases:
            for size in range(3, min(len(phrase), 8) + 1):
                terms.update(
                    phrase[index:index + size]
                    for index in range(len(phrase) - size + 1)
                )
        return {term for term in terms if term not in cls._GENERIC_TERMS}

    @classmethod
    def _has_specific_match(cls, question, content):
        return any(term in content for term in cls._specific_terms(question))

    @staticmethod
    def _keyword_score(question, content):
        """最长连续短语优先，避免泛词出现次数多而抢占结果。"""
        terms = RagService._specific_terms(question)
        matches = [term for term in terms if term in content]
        if not matches:
            return 0

        # 先匹配问题开头的对象或场景，避免“如何处理”等通用后缀抢占结果。
        prefix_matches = [term for term in matches if question.startswith(term)]
        if prefix_matches:
            return 1000 + max(len(term) ** 2 for term in prefix_matches)
        return max(len(term) ** 2 for term in matches)

    def _query_keyword_fallback(self, question):
        ranked = [
            (self._keyword_score(question, item.get("content", "")), item)
            for item in self._load_keyword_chunks()
        ]
        matches = [
            item
            for score, item in sorted(ranked, key=lambda pair: pair[0], reverse=True)
            if score > 0 and self._has_specific_match(question, item.get("content", ""))
        ]
        if not matches:
            return {"answer": "", "source": [], "use_knowledge": False}

        selected = matches[:1]
        context = self._format_local_answer(question, [item["content"] for item in selected])
        sources = list(dict.fromkeys(item["source"] for item in selected))
        return {"answer": context, "source": sources, "use_knowledge": True}

    def add_document(self, file_path: str) -> dict:
        _file_path = Path(file_path)
        suffix = _file_path.suffix.lower()
        loader = None

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(file_path)
            elif suffix == ".docx":
                loader = Docx2txtLoader(file_path)
            elif suffix == ".txt":
                encodings = ["utf-8", "gbk", "gb2315", "latin-1"]
                for enc in encodings:
                    try:
                        loader = TextLoader(file_path, encoding=enc)
                        loader.load()
                        break
                    except Exception as e:
                        default_logger.error(f"编码{enc}失败: {e}")
                        continue
                if loader is None:
                    default_logger.error(f"所有编码都加载失败: {file_path}")
                    return {"success": False, "trunk_count": 0, "message": "无法解析txt文件"}
            elif suffix == ".doc":
                try:
                    loader = TextLoader(file_path, encoding="latin-1")
                    loader.load()
                except Exception as e:
                    default_logger.error(f"doc文件加载失败: {e}")
                    return {"success": False, "trunk_count": 0, "message": "doc文件加载失败"}
            else:
                default_logger.error(f"不支持的文件类型: {suffix}")
                return {"success": False, "trunk_count": 0, "message": f"不支持的文件格式：{suffix}"}

            documents = loader.load()
            if not documents:
                default_logger.error(f"文件加载失败:{file_path}")
                return {"success": False, "trunk_count": 0, "message": "文件加载失败"}

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                default_logger.error(f"文件分割失败:{file_path}")
                return {"success": False, "trunk_count": 0, "message": "文件分割失败"}

            for chunk in chunks:
                chunk.metadata['source'] = file_path
            self._store_keyword_chunks(file_path, chunks)
            if self._ensure_components():
                self.vectorstore.add_documents(chunks)
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": settings.K})
                message = "文档已写入向量知识库和关键词检索库"
            else:
                message = "嵌入模型未就绪，已写入关键词检索库"
            return {
                "success": True,
                "trunk_count": len(chunks),
                "message": message
            }

        except Exception as e:
            default_logger.error(f"添加文档{file_path}失败: {e}")
            return {"success": False, "trunk_count": 0, "message": f"添加文档{file_path}失败: {e}"}

    def query(self, question: str) -> dict:
        if not self._ensure_components():
            default_logger.info("嵌入模型未就绪，使用关键词检索库")
            return self._query_keyword_fallback(question)

        try:
            docs_with_score = self.vectorstore.similarity_search_with_score(question, k=settings.K)
            valid_docs = [
                doc for doc, score in docs_with_score
                if score <= self.distance_threshold
            ]
            default_logger.info(f"向量库原始检索{len(docs_with_score)}个文档，距离过滤后有效知识库片段:{len(valid_docs)}")

            if not valid_docs:
                return {"answer": "", "source": [], "use_knowledge": False}

            best_doc = valid_docs[:1]
            sources = list(
                dict.fromkeys(doc.metadata.get("source", "") for doc in best_doc)
            )
            context = self._format_local_answer(
                question,
                [doc.page_content for doc in best_doc]
            )
            return {"answer": context, "source": sources, "use_knowledge": True}

        except Exception as e:
            default_logger.error(f"查询失败: {e}")
            return self._query_keyword_fallback(question)


rag_service = RagService()
