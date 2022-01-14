# KE-T5-Vision


## 필요 패키지 설치

```bash
    pip install torch
    conda install -c pytorch faiss-gpu # or faiss-cpu
    pip install transformers sentencepiece
```

## 모델 사용 방법

```python
import os
import csv
import logging

import tqdm
import torch
from transformers import AutoTokenizer

from index_scorer import FaissScorerExhaustive
# # GPU ------------------------------------------------
# from index_scorer import FaissScorerExhaustiveMultiGPU
# # ----------------------------------------------------
from modeling_encoder import T5EncoderMean

INDEX_PATH="cc12m_filtered.index"
URL_PATH="cc12m_filtered.tsv"
# # GPU ------------------------------------------------
# FVECS_ROOT="fvecs"
# GPU4MODEL=0
# GPUS4FAISS=[1, 2, 3, 4]
# # ----------------------------------------------------

LANGUAGE_MODEL_PATH="KETI-AIR/ke-t5-base"

ENCODER_PATH="hf_model"
LANGUAGE_ENCODER_PATH=os.path.join(ENCODER_PATH, "language")

TOPK=8

text_tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_PATH)

model = T5EncoderMean.from_pretrained(LANGUAGE_ENCODER_PATH)
# # GPU ------------------------------------------------
# model_device = torch.device('cuda:{}'.format(GPU4MODEL))
# model = model.to(model_device)
# # ----------------------------------------------------
model.eval()

text_query = [
            "바닷가에서 달리는 사람들",
            "화려한 원피스를 입은 젊은 여성",
        ]

text_feature = text_tokenizer(
    text_query, 
    return_tensors="pt", 
    truncation='longest_first', 
    padding=True)

faiss_scorer = FaissScorerExhaustive(
            index_path=INDEX_PATH
        )
# # GPU ------------------------------------------------
# faiss_scorer = FaissScorerExhaustiveMultiGPU(
#            fvec_root=FVECS_ROOT,
#            gpu_list=GPUS4FAISS
#        )
# # ----------------------------------------------------

url_data = [
            item for item in tqdm.tqdm(csv.DictReader(
                open(URL_PATH, "r"), 
                delimiter="\t", 
                quoting=csv.QUOTE_MINIMAL, 
                fieldnames=['path', 'image_url']
            ), desc="loading item...")
        ]


with torch.no_grad():
    outputs = model(
        input_ids=text_feature["input_ids"],
        attention_mask=text_feature["attention_mask"],
    )
    q_vecs = outputs[1]
    q_vecs = q_vecs.numpy()

    # # GPU ------------------------------------------------
    # outputs = model(
    #     input_ids=text_feature["input_ids"].to(model_device),
    #     attention_mask=text_feature["attention_mask"].to(model_device),
    # )
    # q_vecs = outputs[1]
    # q_vecs = q_vecs.cpu().numpy()
    # # ----------------------------------------------------


    scores, indice = faiss_scorer.get_topk(q_vecs, TOPK)

    result_list = []

    for t, score, index in zip( range(len(text_query)), 
                                scores, 
                                indice):
        result = [ {
                "k": k+1,
                "score": s,
                "image_url": url_data[i]["image_url"]
            } for k, s, i in zip(range(TOPK), score, index)]
        result_list.append(result)


    for query, result in zip(text_query, result_list):
                print(f"query: {query}\nresults\n"+'-'*40)
                print(result)
                print('-'*40+'\n\n')

```


<table>
    <tr>
    <td>Query</td>
    <td colspan="3">바닷가에서 달리는 사람들</td>
</tr>
<tr>
    <td>Top 1</td>
    <td>Top 2</td>
    <td>Top 3</td>
    <td>Top 4</td>
</tr>
<tr>
    <td>
        <img height="180" width="180"
            src="https://foundershield.com/wp-content/uploads/2018/08/dockless-scooters-featuredv1.jpg"
            alt="score: 193.66">
    </td>
    <td>
        <img height="180" width="180"
            src="https://image.shutterstock.com/image-photo/bojo-beach-accra-ghana-feb-600w-1039724533.jpg"
            alt="score: 193.20">
    </td>
    <td>
        <img height="180" width="180"
            src="https://t1.thpservices.com/previewimage/gallil/ad1c8e20e31f784c3a961b713afe1ec3/ibr-150331.jpg"
            alt="score: 193.07">
    </td>
    <td>
        <img height="180" width="180"
            src="https://image.shutterstock.com/image-illustration/group-running-girls-sandy-beach-600w-448362406.jpg"
            alt="score: 193.03">
    </td>
</tr>
<tr>
    <td>Top 5</td>
    <td>Top 6</td>
    <td>Top 7</td>
    <td>Top 8</td>
</tr>
<tr>
    <td>
        <img height="180" width="180"
            src="https://t1.thpservices.com/previewimage/gallil/2a1386c34d8052da7048f765d1c91fa8/esy-001148739.jpg"
            alt="score: 193.02">
    </td>
    <td>
        <img height="180" width="180"
            src="https://media.gettyimages.com/photos/competitors-warms-up-prior-to-the-start-of-day-one-of-the-2013-surf-picture-id167014549?s=612x612"
            alt="score: 192.94">
    </td>
    <td>
        <img height="180" width="180"
            src="https://media.gettyimages.com/photos/people-in-the-sun-on-varadero-beach-appearing-in-the-abc-news-special-picture-id1161101714?s=612x612"
            alt="score: 192.90">
    </td>
    <td>
        <img height="180" width="180"
            src="https://media.gettyimages.com/photos/people-on-the-beach-in-cadiz-spain-august-24-2014-picture-id513941055?s=612x612"
            alt="score: 192.84">
    </td>
</tr>
</table>


<table>
    <tr>
    <td>Query</td>
    <td colspan="3">화려한 원피스를 입은 젊은 여성</td>
</tr>
<tr>
    <td>Top 1</td>
    <td>Top 2</td>
    <td>Top 3</td>
    <td>Top 4</td>
</tr>
<tr>
    <td>
        <img height="180" width="180"
            src="https://image.shutterstock.com/image-photo/little-baby-girl-dressed-sari-600w-793538422.jpg"
            alt="score: 184.32">
    </td>
    <td>
        <img height="180" width="180"
            src="https://t1.thpservices.com/previewimage/gallil/41f904c4d0a9153dadc0c2770a136314/esy-047500951.jpg"
            alt="score: 184.05">
    </td>
    <td>
        <img height="180" width="180"
            src="https://t1.thpservices.com/previewimage/gallil/7b449e0e952d54478d0e69e2aa69f9a2/esy-000820218.jpg"
            alt="score: 183.97">
    </td>
    <td>
        <img height="180" width="180"
            src="https://t1.thpservices.com/previewimage/gallil/d651ceef3b208d7a83863d66c73aecfc/esy-025884774.jpg"
            alt="score: 183.82">
    </td>
</tr>
<tr>
    <td>Top 5</td>
    <td>Top 6</td>
    <td>Top 7</td>
    <td>Top 8</td>
</tr>
<tr>
    <td>
        <img height="180" width="180"
            src="https://t1.thpservices.com/previewimage/gallil/6d379501504f795870fe1d4f773eb6a5/wr0791338.jpg"
            alt="score: 183.62">
    </td>
    <td>
        <img height="180" width="180"
            src="https://t1.thpservices.com/previewimage/gallil/4a469b8132decc8c954b30725badcbcf/esy-049063797.jpg"
            alt="score: 183.55">
    </td>
    <td>
        <img height="180" width="180"
            src="https://image.shutterstock.com/image-photo/spanish-woman-wearing-bright-red-600w-134775305.jpg"
            alt="score: 183.52">
    </td>
    <td>
        <img height="180" width="180"
            src="https://image.shutterstock.com/image-vector/frida-kahlo-vector-portrait-mexican-600w-1252162534.jpg"
            alt="score: 183.50">
    </td>
</tr>
</table>


`T5EncoderMean`와 `forward` 대신에 `VisionT5MeanBiEncoder`와 `encode_text`를 사용하실 수 있습니다.
다만 `VisionT5MeanBiEncoder`를 사용하실 경우 ViT 모델도 로드되기 때문에 메모리 사용에 있어 약간 비효율적입니다.


```python
from modeling_encoder import VisionT5MeanBiEncoder

ENCODER_PATH="hf_model"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = VisionT5MeanBiEncoder.from_pretrained(ENCODER_PATH)
model = model.to(model_device)
model.eval()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

q_vecs = model.encode_text({
            "input_ids":text_feature["input_ids"].to(model_device),
            "attention_mask":text_feature["attention_mask"].to(model_device),})
q_vecs = q_vecs.cpu().numpy()

```

## Downloads

### Models


[VisionT5MeanBiEncoder](https://drive.google.com/file/d/1Cq9GldmJz7qpHnJQoXrnA8YD317_3lOz/view?usp=sharing)

### Data


| Model | Data | url data | fvecs | exhaustive index |
| --- | --- | --- | --- | --- |
| VisionT5MeanBiEncoder | CC 12M | [Download](https://drive.google.com/file/d/1gyAWODO70no6RQMuokW8JXEsFdHDJ7mc/view?usp=sharing) | [Download](https://drive.google.com/drive/folders/16yPfEwOjIGMo7kqiu9L4iXQUjJQ_aqDz?usp=sharing) | [Download](https://drive.google.com/file/d/19HsMknZJj43lOCmTXlQ32lIZSm7VFhpL/view?usp=sharing) |
