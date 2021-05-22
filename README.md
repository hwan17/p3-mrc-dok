# p3-mrc-dok

## 코드 실행 방법
* T5 모델을 이용한 Generation Model 학습 및 Inference
    * 학습
    ```bash
    python t5_train.py
    ```
    * Inference: t5_infer.ipynb 실행

## 최종 제출
* Reader Model 학습 방법
```bash
python upload/train.py --warmup_step 300 --lr_scheduler_type cosine_with_restarts \
                            --num_train_epochs 5 --per_device_train_batch_size 32 \
                            --learning_rate 5e-5 --do_train
```

* Inference 방법
