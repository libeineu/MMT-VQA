#! /usr/bin/bash
set -e

batch_size=128
beam=5
src_lang=en
tgt_lang=de
ensemble=10
data_dir=multi30k.en-de.mmt_vqa

gpu=0
who=test   # test1, test2, test3
length_penalty=1.0

tag=mmt_vqa_mae
vision_model_name=facebook/vit-mae-base
use_vlm_text_encoder=0

model_dir=checkpoints/$tag
checkpoint=checkpoint_best.pt
cp ${BASH_SOURCE[0]} $model_dir/translate.sh

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation_$who.log

export CUDA_VISIBLE_DEVICES=$gpu

cmd="fairseq-generate data-bin/$data_dir 
  -s $src_lang -t $tgt_lang --task mmt_vqa  
  --path $model_dir/$checkpoint 
  --gen-subset $who
  --batch-size $batch_size --beam $beam --lenpen $length_penalty 
  --quiet --remove-bpe
  --output $model_dir/hyp.txt
  --image-name-dir data-bin/$data_dir
  --ptm-name $vision_model_name
  --source-sentence-dir data-bin"

if [ $use_vlm_text_encoder -eq 1 ]; then
cmd=${cmd}" --use-vlm-text-encoder "
fi

cmd=${cmd}" | tee "${output}
eval $cmd

python3 rerank.py $model_dir/hyp.txt $model_dir/hyp.sorted

# if [ $task == "multi30k-en-de" ] && [ $who == "test" ]; then
# 	ref=data/multi30k/test.2016.de
# elif [ $task == "multi30k-en-de" ] && [ $who == "test1" ]; then
# 	ref=data/multi30k/test.2017.de
# elif [ $task == "multi30k-en-de" ] && [ $who == "test2" ]; then
# 	ref=data/multi30k/test.coco.de

# elif [ $task == "multi30k-en-fr" ] && [ $who == 'test' ]; then
# 	ref=data/multi30k/test.2016.fr
# elif [ $task == "multi30k-en-fr" ] && [ $who == 'test1' ]; then
# 	ref=data/multi30k/test.2017.fr
# elif [ $task == "multi30k-en-fr" ] && [ $who == 'test2' ]; then
# 	ref=data/multi30k/test.coco.fr
# fi	

# hyp=$model_dir/hyp.sorted
# python3 meteor.py $hyp $ref > $model_dir/meteor_$who.log
# cat $model_dir/meteor_$who.log

# # cal accurary
# python3 cal_acc.py $hyp $who $task