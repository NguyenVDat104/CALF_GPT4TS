export CUDA_VISIBLE_DEVICES="0,1,2"
file_name='ETTh_DINOv3.txt'
seq_len=96
bootstrap_eval=0
model=GPT4TS
pre_lens_h="96 192 336 720"
gpt_loc=0
itt=1

# =====================================================================================================
# Configuration 1: DINO-V3 as Backbone (replaces GPT-2 LLM backbone, keeps PCA Cross-Modal Encoder)
# =====================================================================================================
echo "Running CALF with DINO-V3 Backbone..."
for pred_len in $pre_lens_h; 
do
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1'_'$seq_len'_'$pred_len'_DINOv3_backbone' \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 50 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --backbone dinov3 \
    --encoder pca \
    --dinov3_layers 6 \
    --dinov3_pretrained 1 \
    --distill_loss cosine \
    --task_loss l1 \
    --bootstrap_eval $bootstrap_eval \
    --itr $itt \
    --gpu $gpt_loc \
    --model $model \
    --patience 10 \
    --log_fine_name $file_name
echo '====================================================================================================='
done

# =====================================================================================================
# Configuration 2: DINO-V3 as Input Encoder (replaces Encoder_PCA, keeps GPT-2 Backbone)
# =====================================================================================================
echo "Running CALF with DINO-V3 Encoder..."
for pred_len in $pre_lens_h; 
do
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1'_'$seq_len'_'$pred_len'_DINOv3_encoder' \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 50 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --backbone gpt2 \
    --encoder dinov3 \
    --gpt_layers 6 \
    --bootstrap_eval $bootstrap_eval \
    --itr $itt \
    --gpu $gpt_loc \
    --model $model \
    --patience 10 \
    --log_fine_name $file_name
echo '====================================================================================================='
done

# =====================================================================================================
# Configuration 3: End-to-End DINO-V3 (both Encoder and Backbone powered by DINO-V3)
# =====================================================================================================
echo "Running End-to-End CALF with DINO-V3 Encoder & Backbone..."
for pred_len in $pre_lens_h; 
do
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1'_'$seq_len'_'$pred_len'_DINOv3_end2end' \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 50 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --backbone dinov3 \
    --encoder dinov3 \
    --dinov3_layers 6 \
    --dinov3_pretrained 1 \
    --distill_loss cosine \
    --task_loss l1 \
    --bootstrap_eval $bootstrap_eval \
    --itr $itt \
    --gpu $gpt_loc \
    --model $model \
    --patience 10 \
    --log_fine_name $file_name
echo '====================================================================================================='
done
