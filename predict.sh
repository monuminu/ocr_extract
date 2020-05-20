#!/bin/bash
echo "predicting"
python3 tapas/tapas/run_task_main.py \
    --task="SQA" \
    --output_dir="results" \
    --noloop_predict \
    --test_batch_size=1 \
    --tapas_verbosity="ERROR" \
    --compression_type= \
    --init_checkpoint="tapas_sqa_base/model.ckpt" \
    --bert_config_file="tapas_sqa_base/bert_config.json" \
    --mode="predict" 2> error
echo "prediction complete !!"