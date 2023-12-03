# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch, os
from fairseq.data import FairseqDataset, data_utils
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, AutoModel

logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    query, ans, prev_ans_tokens = None, None, None
    if samples[0].get("query", None) is not None:
        query = merge(
            "query",
            left_pad=left_pad_target,
            pad_to_length=None,
        )
        query = query.index_select(0, sort_order)

    if samples[0].get("ans", None) is not None:
        ans = merge(
            "ans",
            left_pad=left_pad_target,
            pad_to_length=None,
        )
        ans = ans.index_select(0, sort_order)
        prev_ans_tokens = merge(
            "ans",
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
            pad_to_length=None,
        )
        prev_ans_tokens = prev_ans_tokens.index_select(0, sort_order)

    img_tensor = None
    if samples[0].get("img_tensor", None) is not None:
        img_tensor = torch.stack([s["img_tensor"] for s in samples], dim=0)
        img_tensor = img_tensor.index_select(0, sort_order)
    
    clip_input_ids = None
    if samples[0].get("clip_input_ids", None) is not None:
        clip_input_ids = torch.stack([s["clip_input_ids"] for s in samples], dim=0)
        clip_input_ids = clip_input_ids.index_select(0, sort_order)
    
    clip_attention_mask = None
    if samples[0].get("clip_attention_mask", None) is not None:
        clip_attention_mask = torch.stack([s["clip_attention_mask"] for s in samples], dim=0)
        clip_attention_mask = clip_attention_mask.index_select(0, sort_order)
    
    source_sen_list = None
    if samples[0].get("source_sen_item", None) is not None:
        sort_order_list = sort_order.tolist()
        source_sen_list = [samples[i]["source_sen_item"] for i in sort_order_list]

    img_list = None
    if samples[0].get("img_item", None) is not None:
        sort_order_list = sort_order.tolist()
        img_list = [samples[i]["img_item"] for i in sort_order_list]

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            'query': query,
            'prev_ans_tokens': prev_ans_tokens,
            "img_tensor": {'pixel_values': img_tensor} if clip_input_ids is None and clip_attention_mask is None else {'pixel_values': img_tensor, 'input_ids': clip_input_ids, 'attention_mask': clip_attention_mask},
            "img_list": img_list,
            "source_sen_list": source_sen_list
        },
        "target": target,
        'ans': ans,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints.index_select(0, sort_order)

    return batch

def get_img_size(model_name):
    if  "facebook/vit-mae" in model_name:
        config = AutoModel.from_pretrained(model_name).config
        image_size = config.image_size
        
    elif "openai/clip-vit" in model_name:
        config = AutoModel.from_pretrained(model_name).config.vision_config
        image_size = config.image_size

    elif "Salesforce/blip" in model_name:
        config = AutoModel.from_pretrained(model_name).config.vision_config
        image_size = config.image_size

    elif "google/vit" in model_name:
        config = AutoModel.from_pretrained(model_name).config
        image_size = config.image_size

    elif "microsoft/beit" in model_name:
        config = AutoModel.from_pretrained(model_name).config
        image_size = config.image_size

    elif "microsoft/swinv2" in model_name:
        config = AutoModel.from_pretrained(model_name).config
        image_size = config.image_size

    else:
        logger.info('wrong vision model name')
        exit()

    return image_size

class MMTVQALanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,       
        src_sizes,
        src_dict,
        tgt=None,   
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        query_dataset=None,
        ans_dataset=None,
        imgname_list=None,
        ptm_name=None,
        split=None,
        source_sen_list=None,
        use_vlm_text_encoder=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        
        # num_buckets == 0
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

        self.ans_dataset = ans_dataset          
        self.query_dataset = query_dataset      
        self.source_sen_list = source_sen_list 
        
        self.img_tensor = None
        self.clip_input_ids, self.clip_attention_mask = None, None

        logger.info("start load {} image tensor".format(split))
        self.img_list = [Image.open(i).convert('RGB') for i in imgname_list]

        img_size = get_img_size(ptm_name)

        if img_size <= 224:
            if use_vlm_text_encoder == False:    # only image 
                self.processor = AutoImageProcessor.from_pretrained(ptm_name)
                
                os.makedirs(os.path.join('data', ptm_name.split('/')[1]), exist_ok=True)

                if os.path.exists(os.path.join('data', ptm_name.split('/')[1], split+'.pt')):
                    self.img_tensor = torch.load(os.path.join('data', ptm_name.split('/')[1], split+'.pt'))
                else:
                    self.img_tensor = self.processor(images=self.img_list, return_tensors="pt", padding=True)['pixel_values']   # n, 3, 224, 224(swinv2为192)
                    torch.save(self.img_tensor, os.path.join('data', ptm_name.split('/')[1], split+'.pt'))

            else:   
                if 'openai/clip-vit' in ptm_name:
                    self.processor = AutoProcessor.from_pretrained(ptm_name)
                    
                    os.makedirs(os.path.join('data', ptm_name.split('/')[1]), exist_ok=True)

                    if os.path.exists(os.path.join('data', ptm_name.split('/')[1], split+'.img_tensor.pt')) and os.path.exists(os.path.join('data', ptm_name.split('/')[1], split+'.input_ids.pt')) and os.path.exists(os.path.join('data', ptm_name.split('/')[1], split+'.attention_mask.pt')):
                        self.img_tensor = torch.load(os.path.join('data', ptm_name.split('/')[1], split+'.img_tensor.pt'))
                        self.clip_input_ids = torch.load(os.path.join('data', ptm_name.split('/')[1], split+'.input_ids.pt'))
                        self.clip_attention_mask = torch.load(os.path.join('data', ptm_name.split('/')[1], split+'.attention_mask.pt'))
                    else:
                        self.clip_tensor = self.processor(text=self.source_sen_list, images=self.img_list, return_tensors="pt", padding=True)
                        self.clip_input_ids = self.clip_tensor["input_ids"]
                        self.clip_attention_mask = self.clip_tensor["attention_mask"]
                        self.img_tensor = self.clip_tensor["pixel_values"]
                        torch.save(self.img_tensor, os.path.join('data', ptm_name.split('/')[1], split+'.img_tensor.pt'))
                        torch.save(self.clip_input_ids, os.path.join('data', ptm_name.split('/')[1], split+'.input_ids.pt'))
                        torch.save(self.clip_attention_mask, os.path.join('data', ptm_name.split('/')[1], split+'.attention_mask.pt'))
                else:
                    logger.info("{} model does not have a text encoder, please set use_vlm_text_encoder to 0".format(ptm_name))
                    exit()
            
            self.img_list, self.source_sen_list = None, None
        
        else:
            if use_vlm_text_encoder == False:
                self.source_sen_list = None
        
        logger.info("load over!")

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        '''
        MMT task: No ans_item and query_item 
        MMT_VQA task: There are ans_item and query_item 
        
        There are two kinds of pre-training visual language models: 
        1. The images of mae, clip, vit and swin are not large, and the server can eat them full. The input parameters are img_tensor, and the others are None 
        2. The images of blip, beit and bridgetower are large, and the input parameters are img_item and source_sen_item, and the others are None 

        Note: 
        1) When clip is used and use_vlm_text_encoder=1, multi-enter clip_input_ids and clip_attention_mask 
        2) source_sen_item is not None only when blip, bridgetower and use_vlm_text_encoder=1 are used
        '''
        ans_item = self.ans_dataset[index] if self.ans_dataset is not None else None
        query_item = self.query_dataset[index] if self.query_dataset is not None else None
        
        img_tensor = self.img_tensor[index] if self.img_tensor is not None else None
        clip_input_ids = self.clip_input_ids[index] if self.clip_input_ids is not None else None
        clip_attention_mask = self.clip_attention_mask[index] if self.clip_attention_mask is not None else None

        img_item = self.img_list[index] if self.img_list is not None else None
        source_sen_item = self.source_sen_list[index] if self.source_sen_list is not None else None
        
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "query": query_item,
            "ans": ans_item,
            "img_tensor": img_tensor,
            "img_item": img_item, 
            "source_sen_item": source_sen_item,
            "clip_input_ids": clip_input_ids,
            "clip_attention_mask": clip_attention_mask,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
