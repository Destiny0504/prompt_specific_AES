for prompt_id in  4 5 6 7 8
do 
python3 -m script.train_doc_tok_seg \
    --batch_size 32 \
    --dropout 0.1 \
    --lr 6e-5 \
    --epoch 80 \
    --seed 92 \
    --exp_name aes_prompt \
    --dataset_path /home/ljx/Desktop/bro/prompt_specific_AES/data \
    --save_step 20 \
    --prompt_id ${prompt_id} \
    --chunk_sizes "110" \
    --cuda
done

for prompt_id in 2 3 4 5 6 7 8
do 
for test_id in 1 2 3 4 5 6 7 8
do 
python3 -m script.test_doc_tok_seg \
    --batch_size 16 \
    --seed 92 \
    --exp_name aes_prompt \
    --dataset_path /home/ljx/Desktop/bro/prompt_specific_AES/data \
    --save_step 20 \
    --start_checkpoint 3620 \
    --end_checkpoint 3840 \
    --prompt_id ${prompt_id} \
    --test_id ${test_id} \
    --chunk_sizes "110" \
    --result_file "qwk_result.txt" \
    --cuda
done
done  