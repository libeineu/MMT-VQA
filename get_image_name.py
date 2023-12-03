import os

# train
o = open('data-bin/multi30k.en-de.mmt_vqa/train.imgname', 'w', encoding='utf-8')
with open('data/flickr30k/train.txt', 'r', encoding='utf-8') as f:
    for l in f:
        p = os.path.join('flickr30k/flickr30k-images', l.strip())
        o.write(p+'\n')

# val
o = open('data-bin/multi30k.en-de.mmt_vqa/valid.imgname', 'w', encoding='utf-8')
with open('data/flickr30k/val.txt', 'r', encoding='utf-8') as f:
    for l in f:
        p = os.path.join('flickr30k/flickr30k-images', l.strip())
        o.write(p+'\n')

# test
o = open('data-bin/multi30k.en-de.mmt_vqa/test.imgname', 'w', encoding='utf-8')
with open('data/flickr30k/test_2016_flickr.txt', 'r', encoding='utf-8') as f:
    for l in f:
        p = os.path.join('flickr30k/flickr30k-images', l.strip())
        o.write(p+'\n')

# test1 2017
o = open('data-bin/multi30k.en-de.mmt_vqa/test1.imgname', 'w', encoding='utf-8')
with open('data/flickr30k/test_2017_flickr.txt', 'r', encoding='utf-8') as f:
    for l in f:
        p = os.path.join('flickr30k/test2017-images', l.strip())
        o.write(p+'\n')

# test2 coco
o = open('data-bin/multi30k.en-de.mmt_vqa/test2.imgname', 'w', encoding='utf-8')
with open('data/flickr30k/test_2017_mscoco.txt', 'r', encoding='utf-8') as f:
    for l in f:
        p = os.path.join('flickr30k/testcoco-images', l.strip())
        o.write(p+'\n')