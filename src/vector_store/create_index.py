# -*- coding: UTF-8 -*-
import requests


es_url = "http://localhost:9200"

mapping = {
    "mappings": {
        "properties": {
            "content": {
                "type": "text"
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

response = requests.put(f"{es_url}/resumes", json=mapping)
print(response)
