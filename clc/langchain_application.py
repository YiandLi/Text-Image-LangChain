#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: model.py
@time: 2023/04/17
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..

https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html
"""

import os

from duckduckgo_search import ddg, ddg_images
from langchain import LLMChain
from langchain.chains import SequentialChain

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate

# from clc.gpt_service import ChatGLMService
from pydantic import Field

from clc.get_zhipuai_service import ZhiPuChatGLMService
# from clc.source_service import SourceService
from clc.source_service_transformers_pkg import SourceService
from typing import Any, Dict, Optional

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class my_RetrievalQA_call(RetrievalQA):
    llm_service: ZhiPuChatGLMService = Field(None)
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        
        docs = self._get_docs(question)  # 通过自己的 Faiss db 调用得到候选文档
        
        # TODO : 整理图片信息，得到 id 2 caption
        img_captions = []
        txt_doc, img_doc = [], []
        for i, doc in enumerate(docs):
            if doc.metadata['category'] == 'ImagePng':
                img_captions.append(doc.page_content)
                img_doc.append(doc)
            else:
                txt_doc.append(doc)
        
        # TODO : 构造 chain，并且 filter picture by caption
        if len(img_captions) > 0:
            img_captions = "\n".join([f"{i} : '{caption}'" for i, caption in enumerate(img_captions)])
            picture_filter_temple = """
                                    用户的问题描述为: "{question}" ，
                                    当前有一些图片，这些图片的编号和描述的分别是: {img_captions} \n
                                    要求：
                                        请用列表的形式返回仅仅和问题有关系的图片对应的索引，比如 `[0, 1]` ；如果没有相关联的图片，则回复一个空列表 `[]`,
                                        请不要分析和回复无关内容，仅仅回复一个 int 类型的索引列表，比如 `[0, 1]` ，如果没有相关联的图片，则回复一个空列表 `[]` 。\n
                                    回复例子：`[1,2]`, `[2,3]`, `[]` \n
                                    答案：
                                    """
            post_process_temple = """对于一个问题的分析为 {chosen_img_id} , 符合的答案被存储在列表中，请提取出对应的答案列表并且回复，比如 [1,2] ;如果为空列表或者没有合适的答案则返回一个空列表 [] """
            
            picture_filter_chain = LLMChain(llm=self.llm_service,
                                            prompt=PromptTemplate(template=picture_filter_temple,
                                                                  input_variables=["question", "img_captions"]),
                                            output_key="chosen_img_id")
            #
            # chosen_img_id = picture_filter_chain.run(
            #     {"question": question, "img_captions": img_captions})  # 'Royal Linens.'
            
            post_process_chain = LLMChain(llm=self.llm_service,
                                          prompt=PromptTemplate(template=post_process_temple,
                                                                input_variables=["chosen_img_id"]),
                                          output_key="_chosen_img_id"
                                          )
            
            img_chosen_chain = SequentialChain(chains=[picture_filter_chain, post_process_chain],
                                               input_variables=["question", "img_captions"],
                                               output_variables=["_chosen_img_id", "chosen_img_id"],
                                               verbose=True)
            
            chosen_img_id = img_chosen_chain({"question": question, "img_captions": img_captions})
            
            try:
                chosen_img_id = eval(chosen_img_id)
            except:
                chosen_img_id = []
            
            docs = txt_doc + img_doc[chosen_img_id]
        else:
            docs = txt_doc
        
        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )
        
        """
        def _call(
            self,
            inputs: Dict[str, List[Document]],
            run_manager: Optional[CallbackManagerForChainRun] = None,
        ) -> Dict[str, str]:
            _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
            docs = inputs[self.input_key]
            # Other keys are assumed to be needed for LLM prediction
            other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
            output, extra_return_dict = self.combine_docs(
                docs, callbacks=_run_manager.get_child(), **other_keys
            )
            extra_return_dict[self.output_key] = output
            return extra_return_dict
        """
        
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


class LangChainApplication(object):
    def __init__(self, config):
        self.config = config
        self.llm_service = ZhiPuChatGLMService()
        self.source_service = SourceService(config)  # 数据库 + 检索模型
    
    def get_knowledge_based_answer(self, query,
                                   history_len=5,
                                   temperature=0.1,
                                   top_p=0.9,
                                   top_k=4,
                                   web_content='',
                                   chat_history=[]):
        if web_content:
            prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                                已知网络检索内容：{web_content}""" + """
                                已知内容: {context}
                                问题: {question}"""
        else:
            prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                 如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                                 已知内容: {context}
                                 问题: {question}"""
        
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []
        
        self.llm_service.temperature = temperature
        
        knowledge_chain = my_RetrievalQA_call.from_llm(
            llm=self.llm_service,
            retriever=self.source_service.vector_store.as_retriever(
                search_kwargs={"k": top_k}),
            prompt=prompt)
        
        knowledge_chain.llm_service = self.llm_service
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}")
        
        knowledge_chain.return_source_documents = True
        
        result = knowledge_chain({"query": query})
        return result
    
    def get_llm_answer(self, query='', web_content=''):
        if web_content:
            prompt = f'基于网络检索内容：{web_content}，回答以下问题{query}'
        else:
            prompt = query
        result = self.llm_service._call(prompt)
        return result


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
        patterns = ['模型问答']  #
        n_gpus = 1
    
    config = LangChainCFG()
    application = LangChainApplication(config)
    
    result = application.get_knowledge_based_answer('cat cat cat !')
    print(result)
    
    # # TODO : TEST WEB
    # results = ddg('马保国是谁')
    # imgs = ddg_images('马保国是谁', max_results=1)
    # web_content = ''
    # if results:
    #     for result in results:
    #         web_content += result['body']
    
    # result = application.get_llm_answer('马保国是谁')
    # print(result)
