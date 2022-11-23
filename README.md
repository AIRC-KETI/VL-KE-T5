# VL-KE-T5

VL-KE-T5은 [KE-T5](https://github.com/AIRC-KETI/ke-t5)와 [ViT](https://github.com/google-research/vision_transformer)의 임베딩 벡터를 Vision-Language parallel Corpus를 이용하여 정렬한 모델입니다.
영어와 한국어 모두 지원하며, Vision-Language Parallel 데이터 셋들을 Google 번역 API를 이용하여 한국어로 번역한 데이터를 추가적으로 이용하였습니다.

학습에 사용된 Vision-Language Parallel 데이터셋은 다음과 같습니다.


<table>
    <tr>
        <td colspan="5">English</td>
        <td colspan="5">Korean (Translated)</td>
        <td>Korean</td>
    </tr>
    <tr>
        <td>CC 3M</td>
        <td>COCO</td>
        <td>SBU</td>
        <td>Visual Genome</td>
        <td>WIT</td>
        <td>CC 3M</td>
        <td>COCO</td>
        <td>SBU</td>
        <td>Visual Genome</td>
        <td>WIT</td>
        <td>WIT</td>
    </tr>
    <tr>
        <td>2,862,265</td>
        <td>414,113</td>
        <td>772,438</td>
        <td>4,322,358</td>
        <td>3,265,279</td>
        <td>2,862,264</td>
        <td>414,113</td>
        <td>772,438</td>
        <td>4,322,358</td>
        <td>3,265,273</td>
        <td>54,956</td>
    </tr>
</table>


## 필요 패키지 설치

```bash
    pip install torch
    conda install -c pytorch faiss-gpu # or faiss-cpu
    pip install transformers sentencepiece
```

faiss의 자세한 설차 방법은 [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)를 참고하시길 바랍니다.

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

## 메모리와 GPU 갯수에 따른 index scorer 선택과 사용 방법

cc12m 데이터의 인덱스는 약 33Gb 정도가 됩니다.

### 1. 시스템 메모리가 40Gb 이상 있을 경우
```python
from index_scorer import FaissScorerExhaustive

INDEX_PATH="cc12m_filtered.index"
faiss_scorer = FaissScorerExhaustive(
            index_path=INDEX_PATH
        )
```

### 2. 시스템 메모리가 작은 경우
```python
from index_scorer import FaissScorer

INDEX_PATH="cc12m_filtered_OPQ64_256-IVF262144_HNSW32-PQ64.index"
# INDEX_PATH="cc12m_filtered_OPQ192_768-IVF262144_HNSW32-PQ192.index"
faiss_scorer = FaissScorer(
            index_path=INDEX_PATH,
        )
```

### 3. 글로벌 메모리가 40Gb 정도인 GPU가 있을 경우 
```python
from index_scorer import FaissScorerExhaustiveGPU

FVECS_ROOT="fvecs"
GPU_ID=0
faiss_scorer = FaissScorerExhaustiveGPU(
            fvec_root=FVECS_ROOT,
            gpu=GPU_ID
        )
```

### 4. 글로벌 메모리가 10Gb 정도인 GPU가 4개 있을 경우 
```python
from index_scorer import FaissScorerExhaustiveMultiGPU

FVECS_ROOT="fvecs"
GPU_ID_LIST=[0, 1, 2, 3]
faiss_scorer = FaissScorerExhaustiveMultiGPU(
            fvec_root=FVECS_ROOT,
            gpu_list=GPU_ID_LIST
        )
```


## Download

### Models


[VisionT5MeanBiEncoder](https://drive.google.com/file/d/1Cq9GldmJz7qpHnJQoXrnA8YD317_3lOz/view?usp=sharing)

[VisionT5MeanBiEncoder (Language Lock)](https://drive.google.com/file/d/1WNZVYuAva7sq4x5lKmlMYTh2S1FwFVix/view?usp=sharing)

[VisionT5MeanBiEncoder + CC 12M 3 epochs](https://drive.google.com/file/d/1XXkXovgLhxn8nydQequfRid31Ri69l2e/view?usp=sharing)

[VisionT5MeanBiEncoder (norm) + CC 12M 3 epochs](https://drive.google.com/file/d/1mayNQ9DToFAk1ecHyNFVY-MRKK5X5xAi/view?usp=share_link)

[VisionEncoderLanguageDecoder(VELD) + CC 12M 1 epoch](https://drive.google.com/file/d/10VeuC15_CdNJ2HBcfR8O078aJZM6cktj/view?usp=sharing)

### Data


| Model | Data | # of images | url data | fvecs | exhaustive index | OPQ64-256 | OPQ192_768 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VisionT5MeanBiEncoder | CC 12M | 10,793,580 | [(1.67GB)](https://drive.google.com/file/d/1gyAWODO70no6RQMuokW8JXEsFdHDJ7mc/view?usp=sharing) | [(30.96GB)](https://drive.google.com/drive/folders/16yPfEwOjIGMo7kqiu9L4iXQUjJQ_aqDz?usp=sharing) | [(30.88GB)](https://drive.google.com/file/d/19HsMknZJj43lOCmTXlQ32lIZSm7VFhpL/view?usp=sharing) | [(1.04GB)](https://drive.google.com/file/d/1Y1kYpkjYyqdyVTkf6o_VRuytBO0nc9ZE/view?usp=sharing) | [(2.83GB)](https://drive.google.com/file/d/1UR-cuah-n5ssSdTyvuUMBa-3K0PBUKLf/view?usp=sharing) |
| VisionT5MeanBiEncoder | mmcommons | 99,144,306 | [(14.31GB)](https://drive.google.com/file/d/1E98IakJAhOrkwisxYgtkp8K34sFT7i50/view?usp=sharing) | [(284GB)](https://drive.google.com/drive/folders/1USh51znf_uwiXkZh5-3X0bxbz5lD6_UD?usp=sharing) | - | [(6.97GB)](https://drive.google.com/file/d/1q-7dn7yyQc6Q5ientagj_pTZT5MLhUB7/view?usp=sharing) | [(19.29GB)](https://drive.google.com/file/d/1sg8Ylt-Owo1xtMPWUPyd4bhyQwSD9BjV/view?usp=sharing) |
| VisionT5MeanBiEncoder (Language Lock) | CC 12M | 10,793,580 | [(1.67GB)](https://drive.google.com/file/d/1alt2ctw97i5B2mgiw1V31nM1AZ113MgD/view?usp=sharing) | [(33.2GB)](https://drive.google.com/drive/folders/1ypdMe-7ssv08fDm9QgpDBDQe66pIDV-L?usp=sharing) | - | [(1.04GB)](https://drive.google.com/file/d/14ElP5ZLvoyXDuPD4e5YCL26BfmKRYhDk/view?usp=sharing) | [(2.83GB)](https://drive.google.com/file/d/1YY8QiUhbDqND6aHx7U9zPJoT20XM-dgv/view?usp=sharing) |

## Custom image data 사용하기

개인적으로 준비한 이미지를 이용하고 싶으신 경우 아래 블럭과 같이 tsv 포맷으로 이미지와 이미지의 URL을 만들어 줍니다.

(꼭 image URL일 필요는 없습니다. 2번째 칼럼 값이 return 되기 때문에 해당 이미지에 대하여 return하고 싶은 값이면 됩니다.)


test.tsv
```tsv
{path_to_image_0}\t{image_url_0}
{path_to_image_1}\t{image_url_1}
{path_to_image_2}\t{image_url_2}
...
```

`create_index4retriever.py`를 이용하여 index를 만들어 줍니다.


```bash
DATA_PATH=test.tsv
FVECS_OUT_DIR=test_fvecs
ENCODER_PATH=hf_model

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 create_index4retriever.py \
--batch_size 64 \
--batch_write \
--model_cls VisionT5MeanBiEncoder \
--data_path $DATA_PATH \
--fvecs_dir $FVECS_OUT_DIR \
--hf_path $ENCODER_PATH


# # GPU가 하나만 있는 경우
# python create_index4retriever.py \
# --batch_size 64 \
# --batch_write \
# --model_cls VisionT5MeanBiEncoder \
# --data_path $DATA_PATH \
# --fvecs_dir $FVECS_OUT_DIR \
# --hf_path $ENCODER_PATH
```

2번째 칼럼이 image url인 경우 아래와 같이 테스트할 query json 파일을 만듭니다.


test_query.json
```json
[
    "따뜻한 분위기의 카페",
    "화려한 원피스를 입은 젊은 여성",
    "심플한 원피스를 입은 젊은 여성",
    "축구경기를 응원하는 사람들",
    "강아지와 해변을 산책하는 남자",
    "강아지와 해변을 산책하는 여자",
]
```

`retrieve_images.py`를 이용하여 검색된 이미지를 확인합니다.

```bash
DATA_PATH=test.tsv
FVECS_OUT_DIR=test_fvecs
ENCODER_PATH=hf_model
QUERY_PATH=test_query.json
MD_OUT_DIR=md_out

python retrieve_images.py \
--data_path $DATA_PATH \
--fvecs_dir $FVECS_OUT_DIR \
--hf_path $ENCODER_PATH \
--query_path $QUERY_PATH \
--markdown_out $MD_OUT_DIR \
--model_cls VisionT5MeanBiEncoder

```


`--markdown_out`으로 출력된 markdown 파일들을 확인합니다. VS code와 같은 편집기로 확인하면 Preview를 통해 한눈에 볼 수 있습니다.


## Examples

cc12m에서 검색된 이미지 샘플들을 참조하려면 [CC 12M 샘플](samples/samples_cc12m.md)을 참조하세요.

mmcommons에서 검색된 이미지 샘플들을 참조하려면 [exhaustive 샘플](samples/samples_mmcommons.md), [OPQ192-768 샘플](samples/samples_mmcommons_OPQ192_768.md), [OPQ64-256 샘플](samples/samples_mmcommons_OPQ64_256.md)을 참조하세요.

cc12m에서 언어모델을 학습되지 않도록 gradient를 freeze시키고 학습한 모델로 검색된 이미지 샘플들을 참조하려면 [exhaustive 샘플](samples/samples_cc12m_freeze_lm.md), [OPQ192-768 샘플](samples/samples_cc12m_freeze_lm_OPQ192_768.md), [OPQ64-256 샘플](samples/samples_cc12m_freeze_lm_OPQ64_256.md)을 참조하세요.


## Vision Encoder Language Decoder (VELD)

구글의 [Contrastive Captioners](https://arxiv.org/abs/2205.01917)와 같은 방식으로 학습된 Vision Encoder + Language Decoder 구조의 모델입니다.


```python

from modeling_veldt5 import VELDT5Model
from transformers import AutoTokenizer, ViTFeatureExtractor
from PIL import Image

MODEL_PATH = "veld_e1_linear"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH)

images = [Image.open("images/sample.jpg"), Image.open("images/sample2.jpg")]
pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values

model = VELDT5Model.from_pretrained(MODEL_PATH)

outputs = model.generate(
            pixel_values=pixel_values,
            num_beams=4,
            # do_sample=True,
            max_new_tokens=40,
            num_return_sequences=4,
        )

for img_idx, output in enumerate(outputs):
    print("image only {}: ".format(img_idx), tokenizer.decode(output, skip_special_tokens=True))


''' stdout
image only 0:  A baseball player is standing in front of a baseball field.
image only 1:  A baseball player is standing in front of a baseball stadium.
image only 2:  A baseball player is standing in front of a field.
image only 3:  A baseball player is standing in front of a stadium.
image only 4:  해변에 서 있는 사람.
image only 5:  해변에 서 있는 여자
image only 6:  해변에 서 있는 여자.
image only 7:  해변에 서 있는 사람
'''

```


## Acknowledgement

본 연구는 '자기지도 학습에 의한 시각적 상식으로 영상에서 보이지 않는 부분을 복원하는 기술’(2021-0-00537) 및 ‘정서적 안정을 위한 인공지능 기반 공감서비스 기술 개발’(S0316-21-1002)의 지원을 받아 개발되었습니다.

# TODO

- [X] language model gradient freeze 하고 학습하기
- [ ] hard negative sample 만들기
- [ ] 언어 모델 바꾸기 
