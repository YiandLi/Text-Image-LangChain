"""
PAPER: Language Models Can See: Plugging Visual Controls in Text Generation
CODE: https://github.com/yxuansu/MAGIC
"""

# pip3 install simctg -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

# %cd ./MAGIC

from clc.ImageCaption.MAGIC.image_captioning.language_model.simctg import SimCTG
from clc.ImageCaption.MAGIC.image_captioning.clip.clip import CLIP
from PIL import Image
from IPython.display import display  # to display images

import torch
import tqdm
import os
import json


def init_caption_model(language_model_name=r'cambridgeltl/magic_mscoco',
                       clip_model_name=r"openai/clip-vit-base-patch32",
                       clip_model=None, clip_processor=None, clip_tokenizer=None):
    # language_model_name = r'cambridgeltl/magic_mscoco'  # GPT 2 model is both OK ; ths is a pre-trained on COCO version ; u could also choose "gpt2"
    #  # or r"/path/to/downloaded/openai/clip-vit-base-patch32"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load Language Model
    sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
    generation_model = SimCTG(language_model_name, sos_token, pad_token).to(device)
    generation_model.eval()
    
    clip = CLIP(clip_model_name, clip_model, clip_processor, clip_tokenizer).to(device)
    clip.eval()
    
    return clip, generation_model


def get_img_caption(clip, generation_model, image_path: str):
    k, alpha, beta, decoding_len = 45, 0.1, 2.0, 16
    eos_token = '<|endoftext|>'
    sos_token = r'<-start_of_text->'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    start_token = generation_model.tokenizer.tokenize(sos_token)
    start_token_id = generation_model.tokenizer.convert_tokens_to_ids(start_token)
    input_ids = torch.LongTensor(start_token_id).view(1, -1).to(device)
    
    image_instance = Image.open(image_path)
    print('=' * 50)
    display(image_instance)
    output = generation_model.magic_search(input_ids, k,
                                           alpha, decoding_len, beta, image_instance, clip, 60)
    print(output)
    return output


if __name__ == '__main__':
    # img_dir = "clc/zero_shot_ImageCaption/example_images"
    img_dir = "docs"
    caption_path = "docs/img2caption.json"
    
    image_name_list = [i for i in os.listdir(img_dir) if "jpeg" in i]
    
    k, alpha, beta, decoding_len = 45, 0.1, 2.0, 16
    eos_token = '<|endoftext|>'
    sos_token = r'<-start_of_text->'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip, generation_model = init_caption_model()
    
    start_token = generation_model.tokenizer.tokenize(sos_token)
    start_token_id = generation_model.tokenizer.convert_tokens_to_ids(start_token)
    input_ids = torch.LongTensor(start_token_id).view(1, -1).to(device)
    
    basepath_to_caption_dir = {}
    for image_name in tqdm.tqdm(image_name_list):
        image_path = os.path.join(img_dir, image_name)
        image_instance = Image.open(image_path)
        display(image_instance)
        
        output = generation_model.magic_search(input_ids, k,
                                               alpha, decoding_len, beta, image_instance, clip, 100)
        
        basepath_to_caption_dir[image_name] = output
        
        print(output)
        print('=' * 50)
    
    json.dump(basepath_to_caption_dir, open(caption_path, "w"))
