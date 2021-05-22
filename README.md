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
    * 학습
    ```bash
    python t5_train.py --output_dir KETI-AIR/ke-t5-large/outputs --do_train --do_eval
    ```
    * Inference: t5_infer.ipynb 실행