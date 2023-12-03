import torch
import os
import numpy as np
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt

###
# To visualize attention maps, we need files as follows:
# Images(get from multi30k, resized to 224x224):    	'./images/*.jpg'
# Attention maps(saved while translation):     	 	    './checkpoint/*/visualization/*map.pth'
# Src_tokens(saved while translation):        	 	    './checkpoint/*/visualization/*tokens.pth'
# Origin_src_tokens(saved while translating mask0): 	'./origin_tokens/*tokens.pth'
# Translation results(saved while translating):  	    './checkpoint/*/hypo.txt'
# Dictionary of src(saved while bpe):    	 	        './dict.en.txt'
# Image filename(get from multi30k):       	 	        './test_images.txt'
###



def selective_attention_visualization(hyp_path, image_txt_path, dict_en_path, save_path, which, num, who):
    root_path = os.getcwd()

    # Get the translation order from 'hypo.txt'
    translation_order_list = []
    with open(hyp_path, 'r', encoding='utf-8') as translation_order_file:
        for line in translation_order_file:
            translation_order_list.append(int(line.strip().split('\t')[0]))
    # print(translation_order_list)

    # Get image name list from 'test_images.txt'
    test_images_filename_list = []
    with open(os.path.join(image_txt_path), 'r', encoding='utf-8') as test_images_filename_file:
        for line in test_images_filename_file:
            test_images_filename_list.append(line.strip())
    # print(len(test_images_filename_list))

    # Get the dictionary{id: word} from 'dict.en.txt'
    dic_no2word = {0: '<bos>', 1: '<pad>', 2: '<eos>', 3: '<unk>'}
    with open(dict_en_path, 'r', encoding='utf-8') as dict_file:
        for idx, line in enumerate(dict_file):
            dic_no2word[idx+4]  = line.strip().split()[0]


    # Attention map and src_tokens are divided into 8 batches with batch_size=128 in translation
    idx = 0
    for batch in range(num):
        # Get attention maps, src_tokens and origin_tokens
        attn_map_path = os.path.join(save_path, str(batch) + 'map.pth')
        src_tokens_path = os.path.join(save_path, str(batch) + 'tokens.pth')
        # origin_tokens_path = os.path.join(root_path, 'origin_tokens', str(batch) + 'tokens.pth')

        attn_map = torch.load(attn_map_path, map_location=torch.device('cpu'))
        src_tokens = torch.load(src_tokens_path, map_location=torch.device('cpu'))
        # origin_tokens = torch.load(origin_tokens_path, map_location=torch.device('cpu'))
        
        for sent_num in range(attn_map.shape[0]):
            t = translation_order_list[idx]
            idx += 1
            # print(t)
            filename = test_images_filename_list[t]
            # Images for test
            # if filename != '2321764238.jpg':
            #     continue
            # if filename != '327955368.jpg':
            #     continue
            print(filename)

            img = Image.open(os.path.join('flickr30k', which, filename), mode='r')
            plt.figure(filename, figsize=(8, 8))

            for word_num in range(attn_map.shape[1]):
                # Get the attention map for the word
                attn = attn_map[sent_num][word_num].view(7, 7).cpu().numpy()
                # Get the word with the dictionary and src_tokens
                word = src_tokens.cpu().numpy()[sent_num][word_num]
                word = dic_no2word[word]
                # origin_word = origin_tokens.cpu().numpy()[sent_num][word_num]
                # origin_word = dic_no2word[origin_word]

                # Skip '<pad>' and '<eos>'
                if word == '<pad>' or word == '<eos>':
                    continue

                # Show the image
                plt.subplot(math.ceil(attn_map.shape[1] / 4), 4, word_num + 1)
                plt.title(word, fontsize=9)
                # plt.title(word + '-' + origin_word, fontsize=9)
                plt.imshow(img, alpha=1)
                plt.axis('off')

                img_h, img_w = img.size[0], img.size[1]
                attn = cv2.resize(attn.astype(np.float32), (img_h, img_w))
                normed_attn = attn / attn.max()
                normed_attn = (normed_attn * 255).astype('uint8')

                # Show the visual attention map of the word
                plt.imshow(normed_attn, alpha=0.4, interpolation='nearest', cmap='jet')
                plt.axis('off')

                plt.savefig('rb_imgs_mmt/{}/{}.png'.format(who, os.path.basename(filename).replace('.jpg', '')))

            plt.show()


if __name__ == "__main__":
    # model_path  = 'checkpoints/mmt_vqa_mae_final'
    dict_en_path = 'data-bin/multi30k.en-de.mmt/dict.en.txt'

    # coco
    hyp_path = 'checkpoints/mmt_mae_final/hyp_coco.txt'
    image_txt_path = 'data-bin/multi30k.en-de.mmt_vqa/test2.imgname'
    save_path = 'checkpoints/mmt_mae_final/rb_visual_testcoco'
    which='testcoco-images'
    num = 5
    who = 'coco'
    selective_attention_visualization(hyp_path, image_txt_path, dict_en_path, save_path, which, num, who)
    
    # 2018
    hyp_path = 'checkpoints/mmt_mae_final/hyp_2018.txt'
    image_txt_path = 'data-bin/multi30k.en-de.mmt_vqa/test3.imgname'
    save_path = 'checkpoints/mmt_mae_final/rb_visual_test2018'
    which='test2018-images'
    num = 10
    who = '2018'
    selective_attention_visualization(hyp_path, image_txt_path, dict_en_path, save_path, which, num, who)

    # 2017
    hyp_path = 'checkpoints/mmt_mae_final/hyp_2017.txt'
    image_txt_path = 'data-bin/multi30k.en-de.mmt_vqa/test1.imgname'
    save_path = 'checkpoints/mmt_mae_final/rb_visual_test2017'
    which='test2017-images'
    num = 8
    who = '2017'
    selective_attention_visualization(hyp_path, image_txt_path, dict_en_path, save_path, which, num, who)

    # 2016
    hyp_path = 'checkpoints/mmt_mae_final/hyp_2016.txt'
    image_txt_path = 'data-bin/multi30k.en-de.mmt_vqa/test.imgname'
    save_path = 'checkpoints/mmt_mae_final/rb_visual_test2016'
    which='flickr30k-images'
    num = 8
    who = '2016'
    selective_attention_visualization(hyp_path, image_txt_path, dict_en_path, save_path, which, num, who)
