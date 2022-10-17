
import os
import csv
import json
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from index_scorer import FaissScorer
from modeling_encoder import T5EncoderMean

TOKENIZER_PATH = "KETI-AIR/ke-t5-base"
LANGUAGE_ENCODER_PATH = "data/hf_model/language"
FAISS_INDEX_PATH = "data/cc12m_filtered_OPQ64_256-IVF262144_HNSW32-PQ64.index"
URL_DATA_PATH = "data/cc12m_filtered.tsv"

class Service:
    task = [
        {
            'name': "dense_image_retriever",
            'description': 'retrieve images'
        }
    ]

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self.encoder = T5EncoderMean.from_pretrained(LANGUAGE_ENCODER_PATH)
        self.encoder.eval()
        self.faiss_scorer = FaissScorer(
                index_path=FAISS_INDEX_PATH,
                nprobe=16
            )
        self.url_data = [
            item for item in tqdm(csv.DictReader(
                open(URL_DATA_PATH, "r"), 
                delimiter="\t", 
                quoting=csv.QUOTE_MINIMAL, 
                fieldnames=['path', 'image_url']
            ), desc="loading item...")
        ]
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            with torch.no_grad():
                query, k = self.parse_query(content)
                text_feature = self.tokenizer(
                                    query, 
                                    return_tensors="pt", 
                                    truncation='longest_first', 
                                    padding=True)
                
                outputs = self.encoder(
                    input_ids=text_feature["input_ids"],
                    attention_mask=text_feature["attention_mask"],
                )
                q_vecs = outputs[1]
                q_vecs = q_vecs.numpy()
                scores, indice = self.faiss_scorer.get_topk(q_vecs, k)

                result_list = []

                for t, score, index in zip( range(len(query)), 
                                            scores, 
                                            indice):
                    result = [ {
                            "k": k+1,
                            "score": float(s),
                            "image_url": self.url_data[i]["image_url"]
                        } for k, s, i in zip(range(k), score, index)]
                    result_list.append(result)


                return json.dumps(result_list), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400
    
    def parse_query(self, content):
        query = content.get("query", None)
        k = content.get("k", 4)
        if query is None:
            raise ValueError("request must have 'query' field")
        return query, k

