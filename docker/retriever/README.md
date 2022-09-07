# Image retriever

## 필요한 파일들 다운로드

1. Image들을 Vision Transformer로 encoding한 vector값을 OPQ로 압축한 index
2. Image 이름과 url 목록 (1과 순서가 일치해야함)
3. transformer model weights

1의 경우, 메모리에 여유가 있을 경우 다른 인덱스를 써도 무방

`mmcommons100m` 검색을 할 경우

1. mmcommons_filtered_OPQ64_256-IVF262144_HNSW32-PQ64.index
2. mmcommons_filtered.tsv
3. hf_model.zip

`cc12m` 검색을 할 경우

1. cc12m_filtered_OPQ64_256-IVF262144_HNSW32-PQ64.index (Image vector index)
2. cc12m_filtered.tsv
3. hf_model.zip



## 도커 이미지 빌드
```bash
docker build -t keti/image_retriever:v1 .
```

## 도커 컨테이너 런
```bash
DATA_PATH="다운로드한 파일들이 있는 디렉토리 경로 (절대경로)"

docker run --gpus all --rm -it -p 5000:5000 \
-v ${DATA_PATH}:/root/app/data \
--name image_retriever \
keti/image_retriever:v1
```



