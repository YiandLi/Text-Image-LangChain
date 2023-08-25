import json, os

from unstructured.partition.common import get_last_modified_date
from langchain.docstore.document import Document
from langchain.document_loaders.unstructured import UnstructuredBaseLoader
from typing import Any, List


# class MM_doc:
#
#     def __init__(self, mtype, metadata, text=None, image=None, caption=None, extended_caption=None):
#         self.mtype = mtype
#         self.metadata = metadata
#         if mtype == "img":
#             self.image = image
#             self.caption = caption
#             self.extended_caption = extended_caption
#         else:
#             self.text = text


class UnstructuredImageLoader(UnstructuredBaseLoader):
    """Loader that uses unstructured to load files."""
    
    def __init__(
            self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.img2caption = json.load(open("docs/img2caption.json"))
        super().__init__(mode=mode, **unstructured_kwargs)
    
    def _get_elements(self) -> List:
        # from unstructured.partition.auto import partition
        # return partition(filename=self.file_path, **self.unstructured_kwargs)  # 属性有  text + metadata
        
        return [self.file_path]
    
    def _get_metadata(self) -> dict:
        return {"category": "ImagePng",
                "source": self.file_path,
                "last_modified": get_last_modified_date(self.file_path),
                "caption": self.img2caption.get(os.path.basename(self.file_path))}
    
    def load(self):
        """Load file."""
        elements = self._get_elements()
        if self.mode == "elements":
            docs = list()
            for element in elements:
                metadata = self._get_metadata()
                # NOTE(MthwRobinson) - the attribute check is for backward compatibility
                # with unstructured<0.4.9. The metadata attributed was added in 0.4.9.

                docs.append(Document(page_content=metadata['caption'], metadata=metadata))  # page_content 修改了
        elif self.mode == "single":
            metadata = self._get_metadata()
            text = "\n\n".join([str(el) for el in elements])
            docs = [Document(page_content=text, metadata=metadata)]
        else:
            raise ValueError(f"mode of {self.mode} not supported.")
        return docs
