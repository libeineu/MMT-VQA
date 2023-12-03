# Incorporating Probing Signals into Multimodal Machine Translation via Visual Question-Answering Pairs

This repository is the official implement of our EMNLP paper "Incorporating Probing Signals into Multimodal Machine Translation via Visual Question-Answering Pairs" (Findings).



In this paper, we incorporate probing signals into the training process of the Multimodal Machine Translation (MMT) systems through Visual Question-Answer pairs and alleviate the insensitivity of the MMT model to visual features in complete context by enhancing the cross-modal interaction.



We also released the Multi30K-VQA dataset which is designed to achieve the training of the MMT-VQA joint task. We hope that this data set can promote the research on MMT.



## 📑 Contents

- **[Get Started](#Get-Started)**

- **[Training](#Training)**

- **[Testing](#Testing)**

- **[Multi30K-VQA Dataset](#Multi30K-VQA-Dataset)**

- **[Contact](#Contact)**
- **[Citation](#Citation)**



## Get Started

#### Dependencies

```
Python version == 3.6.7
PyTorch version == 1.9.1
sacrebleu version == 1.5.1
vizseq version == 0.1.15
nltk verison == 3.6.4
```

#### Installation

```bash
git clone 
cd fairseq_mmt_vqa
pip install --editable ./
```

## Training

#### 1. Preprocess Data

**Step1: Prepare Multi30K En-De**

We prepare the Multi30K and Flickr30k entities data following the code of the paper: [On Vision Features in Multimodal Machine Translation](https://arxiv.org/pdf/2203.09173.pdf), here is the [repository](https://github.com/libeineu/fairseq_mmt). You should move prepared *multi30k-en2de* and *flickr30k* into *data/*.

*Note*: From [here](https://github.com/libeineu/fairseq_mmt), you can also get more information about how to create masking data and complete probing tasks.

**Step2: Prepare Multi30K-VQA**

Here first download our cleaned data set **Multi30K-VQA** and put it into *data/Multi30k-VQA*. For specific cleaning methods and details please see the paper. 

To preprocess the query and answer data of Multi30K-VQA, we followed the multi30k preprocessing process and used *Moses* to tokenization and clean it.

Firstly, install *subword-nmt* via pip (from PyPI). We use it for BPE:
```bash
pip install subword-nmt
```

Then, preprocess the data following the scripts below:
```bash
echo 'Cloning Moses github repository...'
git clone https://github.com/moses-smt/mosesdecoder.git

# tokenize and bpe for multi30k_vqa data
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

src="query"
tgt="ans"
lang=de

data_dir="data/Multi30k-VQA"

mkdir -p $data_dir/clean
mkdir -p $data_dir/clean_bpe

for l in $src $tgt; do
    cat $data_dir/train.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC | \
        perl $TOKENIZER -threads 8 -l en >> $data_dir/clean/train.$l
done

subword-nmt apply-bpe -c data/multi30k-en2$lang/code < $data_dir/clean/train.query > $data_dir/clean_bpe/train.query
subword-nmt apply-bpe -c data/multi30k-en2$lang/code < $data_dir/clean/train.ans > $data_dir/clean_bpe/train.ans

for l in $src $tgt; do
    cp $data_dir/clean_bpe/train.$l data/multi30k-en2$lang/
done
```

*Run prepare_multi30k_vqa.sh to prepare Multi30K-VQA*.

**Step3: Fairseq-preprocess**

Here we take En-De data processing as an example to show the script. You should use the *srcdict* we provide. 

*Note*: We use the BPE code of the original dataset to segment words find out the elements that do not exist in the original dataset dictionary and add them to the srcdict we provide. You can also perform the above process to generate.

```bash
src='en'
tgt='de'
TEXT=data/multi30k-en2$tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.2018,$TEXT/test.coco \
  --destdir data-bin/multi30k.en-$tgt.mmt_vqa \
  --workers 8 --joined-dictionary \
  --query $TEXT/train.query --ans $TEXT/train.ans \
  --srcdict data/dict_vqa.en2de.txt \
  --task mmt_vqa

# python get_image_name.py
```
*Run preprocess.sh to preprocess mmt_vqa task data.*

#### 2. Training

Here, take the *facebook/vit-mae-base* provided by Huggingface as an example to show the use of the training script. The following are the explanations of several core parameters:

Core script parameters:

- `vision_model` & `ptm-name`: Model name in Huggingface.
- `use_vlm_text_encoder`: Whether to use a text encoder for pre-trained visual language models. 0 is False, 1 is true.
- `arch`: We have completed the exploration experiment on whether different parts of the model are shared. The SA and Decoder without sharing is the best configuration, which is, *transformer_mmt_vqa_2sa_2decoder*.
- `task`: Task name, for our method, you should choose *mmt_vqa*.
- `source-sentence-dir`: For dual tower pre-trained model, like CLIP.


```bash
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

```

*You can run train.sh instead of the scripts above.*

*Note*: To exactly reproduce our results, make sure to use the checkpoint of backone from [huggingface](https://huggingface.co/). As our paper shows, because we use checkpoints from huggingfaces, none of which are fine-tuned on any dataset, the results will be different when we reproduce the selective attention. Here we provide a list of backbone(vision_model_name in shell) in huggingface supported by our code:

```bash
openai/clip-vit-large-patch14
openai/clip-vit-base-patch16
openai/clip-vit-base-patch32

facebook/vit-mae-base
facebook/vit-mae-huge
facebook/vit-mae-large

Salesforce/blip-image-captioning-base
Salesforce/blip-image-captioning-large
Salesforce/blip-itm-base-flickr
Salesforce/blip-itm-large-flickr
Salesforce/blip-vqa-base
Salesforce/blip-vqa-capfilt-large

BridgeTower/bridgetower-base
BridgeTower/bridgetower-large-itm-mlm-itc

google/vit-base-patch16-224
google/vit-large-patch16-224
google/vit-base-patch16-384
google/vit-large-patch16-384
google/vit-large-patch32-384

microsoft/swinv2-base-patch4-window12-192-22k
microsoft/swinv2-large-patch4-window12-192-22k

microsoft/beit-base-patch16-224
microsoft/beit-base-patch16-384
microsoft/beit-large-patch16-224
microsoft/beit-large-patch16-384
```

## Testing

```bash
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
```



## Multi30K-VQA Dataset

This dataset comprises high-quality, task-specific question-answer pairs in a text-image setting, and aims to provide a fertile ground for subsequent research. The generated corpus contains 29,000 unique question-answer pairs, each generated according to our specifications. We envision this dataset as a valuable resource for the community, potentially inspiring future works in this space. 

While, it is non-trivial to create a high-quality QA dataset the MMT model may learn from noise signals. Here we also want to highlight some details when using LLM to generate the Multi30K-VQA dataset. Our prompt underwent over five iterations to mitigate the hallucination issue in which the question-answer pairs do not obey what we pre-defined. Despite these meticulous adjustments, 1,012 of the generated QA pairs fell short of our expectations. To address this, we employed hand-crafted rules to refine 605 of these pairs. For the remaining cases (407), our team conducted sentence-by-sentence annotations to ensure quality (three annotators). Consequently, our final Multi30K dataset comprises 29000 rigorously vetted QA pairs (the same size as the training set). 

In summary, by using the LLM to ask and answer the source of each sample in the Multi30K dataset, after multiple rounds of tuning prompts, we generate parallel visual question-answering style pairs and get the type of each answer to explore the dataset distribution. The distribution of data sets is shown in the following table:

| Type      | Count |
| --------- | ----- |
| Noun      | 5133  |
| Character | 18423 |
| Color     | 5303  |
| Number    | 141   |



You can download the Multi30K-VQA dataset from this link: 

https://drive.google.com/file/d/1yCeEu7CF5WGuWlM29AEtTFzLfsVsuuO7/view?usp=sharing

If the parallel training data of each sample in Multi30K and Multi30K-VQA are merged, each sample contains six parts: source, target, and image form Multi30K, query, answer, and answer_type from Multi30K-VQA. We hope that this dataset can promote the research of MMT and VQA tasks, especially in promoting the sensitivity of multimodal machine translation models to visual features!

## Contact

If you have any questions, please contact this email: truman.yx.zuo@gmail.com.



## Citation

```
@article{zuo2023incorporating,
  title={Incorporating Probing Signals into Multimodal Machine Translation via Visual Question-Answering Pairs},
  author={Zuo, Yuxin and Li, Bei and Lv, Chuanhao and Zheng, Tong and Xiao, Tong and Zhu, Jingbo},
  journal={arXiv preprint arXiv:2310.17133},
  year={2023}
}
```
