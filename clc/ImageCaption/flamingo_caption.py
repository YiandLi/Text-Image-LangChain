import torch
from PIL import Image
from flamingo_mini import FlamingoModel, FlamingoProcessor
import tqdm
import os
import json

print('preparing model...')
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
model = FlamingoModel.from_pretrained("clc/ImageCaption/flamingo-mini")
model.to(device)
model.eval()
processor = FlamingoProcessor(model.config)

# =========
img_dir = "docs"
caption_path = "docs/img2caption.json"
image_name_list = [i for i in os.listdir(img_dir) if "jpeg" in i]

basepath_to_caption_dir = json.load(open(caption_path, "r"))

for image_name in tqdm.tqdm(image_name_list):
    image_path = os.path.join(img_dir, image_name)
    image_instance = Image.open(image_path)  # load_image
    
    output = model.generate_captions(processor, images=[image_instance], device=device, num_beams=5)
    
    if image_name in basepath_to_caption_dir:
        basepath_to_caption_dir[image_name] = basepath_to_caption_dir[image_name] + " ; " + output[0]
    else:
        basepath_to_caption_dir[image_name] = output[0]
    
    print(output)
    print('=' * 50)

json.dump(basepath_to_caption_dir, open(caption_path, "w"))
