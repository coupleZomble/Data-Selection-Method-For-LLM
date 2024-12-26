current_date=$(date +"%Y%m%d_%H%M")
pred_log="/data/home/chenpz/git_clone_project/jupyter_notebook_test_data_selection/log/build_embeddubg_for_anli_r3_train_${current_date}.log"
start_time=$(date +%s)
echo " Pred Start time: $(date -d @$start_time +'%Y-%m-%d %H:%M:%S')" >>${pred_log}

#example
#anli dataset: bert_token_max_len = 360

# /data/home/chenpz/git_clone_project/All_base_model/models--allenai--longformer-base-4096/snapshots/301e6a42cb0d9976a6d6a26a079fef81c18aa895
#/data/home/chenpz/git_clone_project/All_base_model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/44eb4044493a3c34bc6d7faae1a71ec76665ebc6
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python embeddingd_data.py \
    --model_path "BAAI/bge-large-en-v1.5" \
    --data_path "/data/home/chenpz/git_clone_project/jupyter_notebook_test_data_selection/output/anli_can_we_infer_r3_train.json" \
    --save_path "/data/home/chenpz/git_clone_project/jupyter_notebook_test_data_selection/output/anli_r3_train_embedding_noise_500.pkl" \
    --device "cuda:7" \
    --batch_size 600 \
    --max_length 512 \
    --noise False \
    --use_cls True \
    >> ${pred_log}


echo "############pred end###############" >>${pred_log}
echo "pred End time: $(date)" >>${pred_log}
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
min=$(( (duration % 3600) / 60))
echo "Time elapsed: ${hour}  hour $min min " >>${pred_log}







