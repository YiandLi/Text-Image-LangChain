# [Building Image search with OpenAI Clip](https://anttihavanko.medium.com/building-image-search-with-openai-clip-5a1deaa7a6e2
from PIL import Image
import faiss
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# TODO: Model
model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

# TODO: compute image feature
url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
images = [image]
inputs = processor(images=images, return_tensors="pt")
image_features = model.get_image_features(**inputs)
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

# TODO: intialize faiss index
ins_num, ins_hidden_size = image_features.shape
index = faiss.index_factory(f"IVF{min(int(ins_num / 39), round(ins_num ** 0.5))},Flat")
index.train(image_features)
# index.add_with_ids()

# TODO: get Query text embeddings ( single sentence
texts = ["你是哪个蠢蛋 ？"]
inputs = processor(text=texts, padding=True, return_tensors="pt")
text_feature = model.get_text_features(**inputs)
text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)  # normalize

# TODO: inference
index.nprobe = 3
topK = 5
scores, ids = index.search(text_feature, topK)
ids, scores = ids[0], scores[0]

"""
Text  -> 得到图片 ：
    直接搜索 wiki
    text 检索出对应的图片
    Stable Diffusion 得到图片
    
如何应用：
    1。 先检索出来
    2。 VQA 得到  caption ； 然后 text 合并给 LLM input
    
库内部图片：
    para：图片本身，caption，extended caption
"""
