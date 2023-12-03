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