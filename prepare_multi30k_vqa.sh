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