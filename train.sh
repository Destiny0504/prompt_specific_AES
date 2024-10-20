SEED=927

for fold in 3 4 2
do 
python3 -m script.train_mixed_regression \
    --batch_size 4 \
    --dropout 0.1 \
    --lr 2e-5 \
    --smooth_factor 0.1 \
    --tau 0.1 \
    --epoch 30 \
    --seed $SEED \
    --model_name microsoft/deberta-v3-base \
    --dataset_path /scratch1/jt/test/data/fold_${fold}/mixed.pk \
    --exp_name aes_regression_mixed_new_dataset_fold_${fold}_no_supervised \
    --accumulation_step 16 \
    --start_log 70000 \
    --save_step 100 \
    --gpu 0

for prompt_id in 1 2 3 4 5 6 7 8
do 
CUDA_VISIBLE_DEVICES=0,1 python3 -m script.test_regression \
    --batch_size 64 \
    --start_checkpoint 70000 \
    --end_checkpoint 75000 \
    --checkpoint_step 100 \
    --exp_name aes_regression_mixed_new_dataset_fold_${fold}_no_supervised \
    --model_name microsoft/deberta-v3-base \
    --dataset_path  /scratch1/jt/test/data/fold_${fold}/test_${prompt_id}.pk \
    --gpu 0
done
done

for fold in 3 4 2
do 
python3 -m script.train_mixed_regression \
    --batch_size 4 \
    --dropout 0.1 \
    --lr 2e-5 \
    --smooth_factor 0.1 \
    --tau 0.1 \
    --epoch 30 \
    --seed $SEED \
    --model_name microsoft/deberta-v3-base \
    --dataset_path /scratch1/jt/test/data/fold_${fold}/mixed.pk \
    --exp_name aes_regression_mixed_new_dataset_fold_${fold}_no_classification \
    --accumulation_step 16 \
    --start_log 70000 \
    --save_step 100 \
    --supervise_con_on_cls \
    --gpu 0

for prompt_id in 1 2 3 4 5 6 7 8
do 
CUDA_VISIBLE_DEVICES=0,1 python3 -m script.test_regression \
    --batch_size 64 \
    --start_checkpoint 70000 \
    --end_checkpoint 75000 \
    --checkpoint_step 100 \
    --exp_name aes_regression_mixed_new_dataset_fold_${fold}_no_classification \
    --model_name microsoft/deberta-v3-base \
    --dataset_path  /scratch1/jt/test/data/fold_${fold}/test_${prompt_id}.pk \
    --gpu 0
done
done
