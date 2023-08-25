#!/usr/bin/env python
# -*- coding:utf-8 _*-

import json
import zhipuai

from typing import Dict, Union, Optional
from typing import List

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

zhipuai.api_key = "c449fc56df40c2cb296568077581360e.nAkSflf4Ok2F8KjI"

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 请求模型
class ZhiPuChatGLMService(LLM):
    temperature: float = 0.2
    history = []
    model: object = "chatglm_std"  # chatglm_pro , chatglm_std , chatglm_lite
    
    def __init__(self):
        super().__init__()
    
    @property
    def _llm_type(self) -> str:
        return "ZhiPu: " + self.model
    
    def _call(self,
              prompt,
              stop: Optional[List[str]] = None
              ) -> str:
        self.history.append(
            {"role": "user", "content": prompt}
        )
        
        response = zhipuai.model_api.invoke(
            model=self.model,
            temperature=self.temperature,
            prompt=self.history
        )
        
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        
        answer = json.loads(response['data']['choices'][0]['content'])
        self.history.append(
            {"role": "assistant", "content": answer}
        )
        
        return answer


if __name__ == '__main__':
    zhipuLLM = ZhiPuChatGLMService()
    print(zhipuLLM("你好，上海为什么物价那么高？"))
    
    # response = zhipuai.model_api.invoke(
    #     model='chatglm_lite',
    #     # prompt=self.history
    #     prompt=[  # 按照 {"role": "user" / "assistant", "content": "你好"} 的键值对形式进行传参
    #         {"role": "user", "content": "你好"},
    #         {"role": "assistant", "content": "我是人工智能助手"},
    #         {"role": "user", "content": "你叫什么名字"},
    #         {"role": "assistant", "content": "我叫chatGLM"},
    #         {"role": "user", "content": "你都可以做些什么事"},
    #     ]
    # )
    #
    # print(response)
    # """
    #
    # {'code': 200, 'msg': '操作成功',
    #  'data': {
    #         'request_id': '7830525412781883989',
    #         'task_id': '7830525412781883989',
    #         'task_status': 'SUCCESS',
    #         'choices': [
    #                      {  'role': 'assistant',
    #                         'content': '" 我是一台大型语言模型，可以进行自然语言处理和生成，以下是一些我可以完成的事情：\\n\\n1. 对话：我们可以进行文字聊天，讨论各种话题，我会尽力理解您的问题并给出相应的回答。\\n\\n2. 文本生成：我可以根据您提供的主题或关键字生成文章、段落或句子，帮助快速获取所需的文本内容。\\n\\n3. 翻译：如果需要将一种语言的文本翻译成另一种语言，我可以为您提供自动翻译服务。目前我支持多种语言之间的翻译。\\n\\n4. 提供知识：我可以回答各种问题，并提供相关的知识和信息，帮助快速获取所需的知识和帮助。\\n\\n5. 机器学习：我还可以用于机器学习任务，例如生成训练数据、预测结果或分类数据。\\n"'
    #                         }],
    #         'usage': {'total_tokens': 163}}, 'success': True}
    #
    # """
