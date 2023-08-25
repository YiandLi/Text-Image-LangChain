#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: search.py
@time: 2023/04/17
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

import uuid

import os, torch
import numpy as np
from duckduckgo_search import ddg
from langchain import InMemoryDocstore
from langchain.document_loaders import UnstructuredFileLoader
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import dependable_faiss_import
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from typing import Any, List, Optional
from clc.doc_cls import UnstructuredImageLoader
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class myFAISS(FAISS):
    
    @classmethod
    def __from(
            cls,
            texts: List[str],
            embeddings: List[List[float]],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> FAISS:
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatIP(len(embeddings[0]))  # 修改为 IndexFlatIP ，使用点积相似度
        index.add(np.array(embeddings, dtype=np.float32))
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        index_to_id = {i: str(uuid.uuid4()) for i in range(len(documents))}
        docstore = InMemoryDocstore(
            {index_to_id[i]: doc for i, doc in enumerate(documents)}
        )
        return cls(embedding.embed_query, index, docstore, index_to_id, **kwargs)


class SourceService(object):
    def __init__(self, config):
        self.vector_store = None
        self.config = config
        self.docs_path = self.config.docs_path
        self.vector_store_path = self.config.vector_store_path
        
        self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").eval()
        self.processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        
        self.init_source_vector()
    
    def get_img_emb(self, img_list):
        with torch.no_grad():
            inputs = self.processor(images=img_list, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
            return image_features.cpu().numpy().tolist()[0]  # 返回一个样本的 hidden_size - d vector
    
    def get_txt_emb(self, txts):
        with torch.no_grad():
            inputs = self.processor(text=txts, padding=True, truncation=True, max_length=self.model.config.max_length,
                                    return_tensors="pt")
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize
            return text_features.cpu().numpy().tolist()[0]  # 返回一个样本的 hidden_size - d vector
    
    def init_source_vector(self):
        """
        初始化本地知识库向量
        :return:
        """
        txt_docs, img_docs = [], []
        
        # TODO: Read Data
        for doc in os.listdir(self.docs_path):
            if doc.endswith('.txt'):
                print(f"Loading {doc}  ... ")
                loader = UnstructuredFileLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                txt_docs.extend(doc)
            
            elif doc.endswith("png") or doc.endswith("jpg") or doc.endswith("jpeg"):
                print(f"Loading {doc}  ... ")
                loader = UnstructuredImageLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                img_docs.extend(doc)
        
        # TODO: Embed
        txt_metadatas = [d.metadata for d in txt_docs]
        text_embeddings = [(d.page_content, self.get_txt_emb(d.page_content)) for d in txt_docs]
        
        img_metadatas = [d.metadata for d in img_docs]
        img_embeddings = [(d.page_content, self.get_img_emb(Image.open(d.page_content))) for d in img_docs]
        
        # TODO: Save To DB ； 不需要 shuffle 的
        self.model.embed_query = self.get_txt_emb  # 构建 FAISS 需要调用这个方法 ，用来 emb 后续的 query
        self.vector_store = myFAISS.from_embeddings(text_embeddings + img_embeddings,
                                                    self.model,
                                                    txt_metadatas + img_metadatas)
        
        # self.vector_store = FAISS.from_documents(docs, self.model)
        self.vector_store.save_local(self.vector_store_path)
    
    def add_document(self, document_path):
        loader = UnstructuredFileLoader(document_path, mode="elements")
        doc = loader.load()
        self.vector_store.add_documents(doc)
        self.vector_store.save_local(self.vector_store_path)
    
    def load_vector_store(self, path):
        if path is None:
            self.vector_store = myFAISS.load_local(self.vector_store_path, self.model)
        else:
            self.vector_store = myFAISS.load_local(path, self.model)
        return self.vector_store
    
    def search_web(self, query):
        
        # SESSION.proxies = {
        #     "http": f"socks5h://localhost:7890",
        #     "https": f"socks5h://localhost:7890"
        # }
        try:
            results = ddg(query)
            web_content = ''
            if results:
                for result in results:
                    web_content += result['body']
            return web_content
        except Exception as e:
            print(f"网络检索异常:{query}")
            return ''


if __name__ == '__main__':
    class LangChainCFG:
        llm_model_name = "chatglm_lite"
        embedding_model_name = 'GanymedeNil/text2vec-base-chinese'  # 检索模型文件 or hugging face 远程仓库
        vector_store_path = './cache'
        docs_path = './docs'
        kg_vector_stores = {
            '中文维基百科': './cache/zh_wikipedia',
            '大规模金融研报': './cache/financial_research_reports',
            '初始化': './cache',
        }  # 可以替换成自己的知识库，如果没有需要设置为None
        # kg_vector_stores=None
        patterns = ['模型问答', '知识库问答']  #
        n_gpus = 1
    
    
    config = LangChainCFG()
    
    source_service = SourceService(config)
    search_result = source_service.vector_store.similarity_search_with_score('animal', k=5)
    print(search_result)
    
    search_result = source_service.vector_store.similarity_search_with_score_by_vector(
        source_service.get_img_emb(Image.open("docs/cat1.jpeg")))
    print(search_result)
    
    # source_service.add_document('/home/searchgpt/yq/Knowledge-ChatGLM/docs/added/科比.txt')
    # search_result = source_service.vector_store.similarity_search_with_score('可爱的猫猫，不是人', k=5)
    # print(search_result)
    #
    # vector_store = source_service.load_vector_store()
    # search_result = source_service.vector_store.similarity_search_with_score('科比')
    # print(search_result)
