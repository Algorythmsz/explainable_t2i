# explainable_t2i

CLIP 기반 text-to-image retrieval 베이스라인과 평가 코드를 담은 프로젝트입니다.

## 프로젝트 구조

- `data/data.py`
  - 메인 CLI 엔트리포인트입니다.
  - `build-index`, `eval`, `topk` 명령을 받아 전체 파이프라인을 실행합니다.

- `data/dataset_io.py`
  - 데이터셋 입출력/전처리 모듈입니다.
  - `captions.txt`를 읽어 `(image, caption)` 쌍으로 파싱하고,
    이미지별 캡션 매핑(`image -> captions`)과 train/val/test split을 만듭니다.

- `data/clip_model.py`
  - CLIP 모델 관련 모듈입니다.
  - Hugging Face에서 CLIP 모델/프로세서를 로드하고,
    이미지/텍스트 임베딩을 배치 단위로 생성합니다.

- `data/index_io.py`
  - 인덱스 저장/불러오기 모듈입니다.
  - 생성된 이미지 임베딩, 이미지 이름, split 정보, 메타데이터를 파일로 저장하고,
    추후 평가/검색 시 재사용할 수 있도록 다시 로드합니다.

- `data/retrieval.py`
  - 검색/평가 로직 모듈입니다.
  - 코사인 유사도 기반 Top-k 검색을 수행하고,
    `Recall@K` 지표를 계산합니다.

- `data/utils.py`
  - 공통 유틸리티 모듈입니다.
  - 랜덤 시드 고정(`seed_everything`)으로 실험 재현성을 맞춥니다.

- `data/flickr30k/captions.txt`
  - Flickr30k 캡션 메타 파일입니다.
  - 각 이미지 파일명과 캡션 문장이 CSV 형식으로 저장되어 있습니다.

- `data/flickr30k/Images/` (gitignore 대상)
  - 실제 Flickr30k 이미지 파일들이 위치하는 폴더입니다.

## 실행 흐름

1. `build-index`
   - 이미지 임베딩 인덱스를 생성하고 디스크에 저장
2. `eval`
   - 저장된 인덱스를 불러와 split별 `Recall@K` 평가
3. `topk`
   - 임의 text query에 대한 상위 k개 이미지 검색

## 기본 실행 예시

```bash
python3 data/data.py build-index \
  --images-dir data/flickr30k/Images \
  --captions-file data/flickr30k/captions.txt \
  --output-dir artifacts/flickr30k_clip

python3 data/data.py eval \
  --index-dir artifacts/flickr30k_clip \
  --split val --k 1 5 10

python3 data/data.py topk \
  --index-dir artifacts/flickr30k_clip \
  --query "a dog running through the grass" \
  --k 5
```
