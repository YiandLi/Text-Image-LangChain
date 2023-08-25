import faiss
import torch

sentence_embeddings = torch.randn((150, 16))

dimension = sentence_embeddings.shape[1]
quantizer = faiss.IndexFlatL2(dimension)
nlist = 50
m = 2
nbits_per_idx = 2
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits_per_idx)

index.train(sentence_embeddings)
index.add(sentence_embeddings)
index.nprobe = 3

import time

topK = 5
search = torch.randn((1, 16))

costs = []
for x in range(10):
    t0 = time.time()
    D, I = index.search(search, topK)
    t1 = time.time()
    costs.append(t1 - t0)

print("平均耗时 %7.3f ms" % ((sum(costs) / len(costs)) * 1000.0))
