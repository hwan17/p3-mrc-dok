# p3-mrc-dok

## 최종 제출 파일
* 실제 성능 (최종 리더보드 기준)
     * EM: 55.56%
     * F1: 66.45%

* Reader Model 학습 방법
```bash
python upload/train.py --warmup_step 300 --lr_scheduler_type cosine_with_restarts \
                            --num_train_epochs 5 --per_device_train_batch_size 32 \
                            --learning_rate 5e-5 --do_train
```

* Inference 방법
    * `code/elasticsearch.ipynb` 실행
    * `python upload/inference.py --model_path_or_dir model_path --output_dir output_dir`
    * `code/post_processing_ranking.ipynb`에서 파일 경로 수정 후 실행
    
    
## 기타 시도
* T5 모델을 이용한 Generation Model 학습 및 Inference
    * 학습 방법
    ```bash
    python t5_train.py --output_dir KETI-AIR/ke-t5-large/outputs --do_train --do_eval
    ```
    * Inference: t5_infer.ipynb 실행
        * EM: 55.42%
        * F1: 68.07%

---

# MRC
- LB 점수: 55.56%
- 순위: 42등

## 검증(Validation) 전략

대회에서 사용한 검증 전략은 다음과 같습니다:

- 제공된 klue mrc 데이터셋에서 train과 validation 셋에 모두 등장하는 context를 train 데이터로 재배치하였습니다.
- 유니크한 context만 있는 validation set을 구성하여 실험 및 검증을 수행하였습니다.

## 사용한 모델 아키텍처 및 하이퍼 파라미터

아래는 사용한 모델 아키텍처 및 하이퍼 파라미터에 대한 정보입니다:

- 모델 아키텍처: monolgg/koelectra-base-v3-discriminator
- LB 점수: 55.83%

## 추가 전처리

- Unique한 validation을 구성하기 위해 데이터를 재구성하였습니다.

## 추가 시도

1. Tfidf의 maxfeatures 크기를 실험적으로 조정하여 성능 향상을 시도하였습니다.
   - maxfeatures 크기를 50000 -> 100000 -> 200000으로 키워 보았습니다. 결과적으로 성능 향상을 확인하였습니다. (14.58% -> 32.58%)

2. BM25를 적용하여 retrieval을 수행하였습니다.
   - BM25를 적용한 결과, EM (Exact Match) 점수가 50% 이상 향상되었습니다.

3. DPR에서 max_length가 긴 문서의 의미를 다 담지 못하는 것으로 판단하여 길이가 긴 context를 잘라주면서 데이터셋을 재구성하여 학습하였습니다.
   - DPR 기본 EM 점수를 12%에서 30%로 향상시켰습니다.

4. Elastic search를 적용하였습니다.
   - BM25 적용 결과가 51.25%에서 55.42%로 성능이 향상되었습니다.

5. Cosine with restart를 활용한 scheduler 적용
   - Scheduler에 Cosine with restart와 warmup step 300을 적용하여 성능을 개선하였습니다.

6. Top k를 변화시키면서 답을 선택하는 방법 시도
   - Top k 값을 5, 10, 15, 20 등으로 변경하여 답을 선택하는 실험을 수행하였습니다.

7. 다양한 데이터셋 활용
   - Klue 데이터셋이 상대적으로 작은 크기를 가지고 있어서 squad-kor 데이터를 train 데이터에 추가하여 학습을 진행하였습니다.

## 추가 후처리

- 다음과 같은 후처리 작업을 수행하였습니다:
  - 부사 제거
  - 조사 탈락
  - 정답 길이가 15자 이상일 때 다음 정답을 채택

## 앙상블 방법

- 앙상블 방법으로 Hard Voting을 적용하였습니다.
- 여러 모델 중에서 정답으로 많이 출현한 답을 채택하며, 횟수가 동일한 경우에는 EM이 더 높은 모델의 정답을 선택하였습니다.

## 시도했으나 잘 되지 않았던 것들

1. DPR의 성능을 향상시키기 위해 문맥을 자르고, 정답을 반드시 포함하도록 전처리를 시도하였으나 성능 향상을 이루지 못했습니다.

2. Xlm-roberta 모델에서 후처리를 시도해보지 않았습니다. 이를 통해 EM이 향상될 수 있는지 확인하지 못했습니다.

3. Generation 모델을 사용해보려 했으나 학습 시간이 오래 걸려 결과 확인을 못했습니다.

4. Ranking

과 Elastic search에서 파라미터 조정이 부족했고, 다른 팀과 비교했을 때 성능 향상 폭이 적었습니다.

5. DPR에서 p, q encoder를 bert-base-multilingual-cased 모델 외에 kobert, bert-kor-base 등과 같은 구조의 모델로 학습시켜 보았으나, 기본 모델에 비해 성능이 떨어졌습니다.

6. Fp16 옵션을 사용하여 배치 크기를 키울 수 있었지만, scheduler에 영향을 주는 파라미터가 에폭 단위로 다르게 작동해 성능 향상을 이루지 못했습니다.

7. DPR에서 p, q encoder를 bert model 대신 electra 모델로 시도하였으나 출력 모양이 다르게 되어 실패했습니다.

8. BM25를 기반으로 선택한 hard negative를 추가했을 때 성능이 향상되지 않고 소폭 하락했습니다.

## 학습 과정에서의 교훈

- 모델의 파라미터 튜닝 외에도 데이터 전처리 및 후처리 작업이 중요하다는 것을 깨달았습니다.
- 논문의 모델 구조를 이해하기 어려울 때 코드를 통해 이해할 수 있는 방법을 발견하였습니다.
- 학습 시 배치 사이즈를 키우는데 Fp16 옵션을 사용할 수 있다는 것을 배웠습니다.
- 학습 과정에서 모델의 구조나 출력을 수정하여 원하는 기능을 만들어내는 방법을 연구하겠습니다.

## 한계와 도전 숙제

- DPR의 성능을 향상시키기 위한 실험과 모델 변경을 시도하겠습니다.
- Multilingual model의 출력을 후처리하는 방법을 학습하여 적용해보겠습니다.
- Fp16 옵션을 활용하여 모델 학습 시간을 단축하고, 다양한 파라미터에 대한 실험을 진행하겠습니다.
- 데이터셋 크기가 작을 때 앙상블 방법의 merge 방식을 개선해보겠습니다.
- 모델의 구조나 출력을 수정하여 원하는 기능을 만들어내는 방법을 연구하고 적용해보겠습니다.