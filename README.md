
## Description
This is a project based on LangChain, together with multi-modal models, aiming at understand, save, retrieve or even create images in the (retrieval-augmented) dialogue system.

There have been so many great works based on LangChain (Thx for developers). However, when it comes to image-type documents, many limitations still exist, including:
- Reading users' document, here I mean the Emoticon-Packs-liked types, instead of PDF documents **-> VQA**
- Rave image and retrieve images into and from DB **-> universal text-image retrieval**
- Based on retrieved image, how to combine  the information with the text, since there is a huge information and modal boundary between txt and img.
- How to return pictures? directly return the retrieved ones, or  make modification on them, or generation (been supported by LangChain) and How ?  
- ...

<!-- 
## 🔥 Demo shown 
--> 


## 🚀 Run it
1. save all the text or image documents into the folder `docs`
2. run `clc/ImageCaption/MAGIC` or `clc/ImageCaption/flamingo_caption.py` to get image caption caches (an off-line method).
U can both run them, and the two different captions will be combined and saved. 
3. run `clc/langchain_application.py` for a query test.

## ⚡️ features
1. Save image documents in Faiss DB , the representation is $e_\text{text}  + \alpha_\text{ratio} *  e_\text{iamge} $
2. Support text query input for multi-modal documents retrieval
3. Support answer generation based on image captions and text , there is still an re-rank operations to balance the bad performance of image representation. 


## 🔨 TODO
- [ ] Try much more powerful model : in the project, I used Flamingo-mini and Clip-16 , however the performance sometimes is not good
- [ ] Read users' image
- [ ] Gradio demonstration
<!-- - [x] --> 
...

## Contact
mail @ **Yliu8258@usc.edu**

## Reference
- [Chinese-LangChain](https://github.com/yanqiangmiffy/Chinese-LangChain)
- [Language Models Can See: Plugging Visual Controls in Text Generation](https://github.com/yxuansu/MAGIC)
- [flamingo-mini](https://github.com/dhansmair/flamingo-mini)

## Citation
<details> 
   
        ```
        @article{su2022language,
          title={Language Models Can See: Plugging Visual Controls in Text Generation},
          author={Su, Yixuan and Lan, Tian and Liu, Yahui and Liu, Fangyu and Yogatama, Dani and Wang, Yan and Kong, Lingpeng and Collier, Nigel},
          journal={arXiv preprint arXiv:2205.02655},
          year={2022}
        }
        @article{su2022contrastive,
          title={A Contrastive Framework for Neural Text Generation},
          author={Su, Yixuan and Lan, Tian and Wang, Yan and Yogatama, Dani and Kong, Lingpeng and Collier, Nigel},
          journal={arXiv preprint arXiv:2202.06417},
          year={2022}
        }
        @article{Alayrac2022Flamingo,
            title   = {Flamingo: a Visual Language Model for Few-Shot Learning},
            author  = {Jean-Baptiste Alayrac et al},
            year    = {2022}
        }
        ```
</details> 
# Text-Image-LangChain
