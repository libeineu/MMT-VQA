#! /usr/bin/bash
set -e

SA_attention_dropout=0.1
SA_image_dropout=0.1
SA_text_dropout=0
vqa_SA_attention_dropout=0.1
vqa_SA_image_dropout=0.1
vqa_SA_text_dropout=0

device=0
gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`
# gpu_num=1

tag=mmt_vqa_mae
src_lang=en
tgt_lang=de
vision_model_name=facebook/vit-mae-base
use_vlm_text_encoder=0

save_dir=checkpoints/$tag
keep_last_epochs=10
criterion=label_smoothed_cross_entropy_mmt_vqa

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

data_dir=multi30k.en-de.mmt_vqa
arch=transformer_mmt_vqa_2sa_2decoder

fp16=1
lr=0.005
warmup=2000
max_tokens=4096
update_freq=1
dropout=0.3
weight=0.3

cp ${BASH_SOURCE[0]} $save_dir/train.sh

cmd="fairseq-train data-bin/$data_dir
  --save-dir $save_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --dropout $dropout
  --criterion $criterion --label-smoothing 0.1
  --task mmt_vqa
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens --update-freq $update_freq
  --share-all-embeddings
  --find-unused-parameters
  --skip-invalid-size-inputs-valid-test
  --patience $keep_last_epochs
  --keep-last-epochs $keep_last_epochs
  --image-name-dir data-bin/$data_dir
  --ptm-name $vision_model_name
  --vision-model $vision_model_name
  --weight $weight
  --source-sentence-dir data-bin"

if [ $use_vlm_text_encoder -eq 1 ]; then
cmd=${cmd}" --use-vlm-text-encoder "
fi
if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi
if [ -n "$SA_image_dropout" ]; then
cmd=${cmd}" --SA-image-dropout "${SA_image_dropout}
fi
if [ -n "$SA_text_dropout" ]; then
cmd=${cmd}" --SA-text-dropout "${SA_text_dropout}
fi
if [ -n "$SA_attention_dropout" ]; then
cmd=${cmd}" --SA-attention-dropout "${SA_attention_dropout}
fi
if [ -n "$vqa_SA_image_dropout" ]; then
cmd=${cmd}" --vqa-SA-image-dropout "${vqa_SA_image_dropout}
fi
if [ -n "$vqa_SA_text_dropout" ]; then
cmd=${cmd}" --vqa-SA-text-dropout "${vqa_SA_text_dropout}
fi
if [ -n "$vqa_SA_attention_dropout" ]; then
cmd=${cmd}" --vqa-SA-attention-dropout "${vqa_SA_attention_dropout}
fi

export CUDA_VISIBLE_DEVICES=$device

if [ -f $save_dir/train.log ]; then
    echo "File $save_dir/train.log exists!"
    exit
fi

cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
