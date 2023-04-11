import os

import torch
from transformers import BertTokenizer
from kocharelectra.tokenization_kocharelectra import KoCharElectraTokenizer
from model.transformer import Transformer
from dataset import TranslationDataset
from torch.autograd import Variable
from model.util import subsequent_mask
import nltk.translate.bleu_score as bleu


if __name__=="__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    vocab_path = 'data/wiki-vocab.txt'
    ko_vocab_path = 'kocharelectra/vocab.txt'
    data_path =   data_path = 'data/구어체(1).csv'
    checkpoint_path = 'checkpoints'

    tokenizer1 = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
    tokenizer2 = KoCharElectraTokenizer(vocab_file=ko_vocab_path, do_lower_case=False)

    model_name = 'transformer-translation-spoken145400'
    vocab_num = 22000
    max_length = 100
    d_model = 768
    head_num = 8
    dropout = 0.1
    N = 6

    model = Transformer(en_vocab_num=vocab_num,
                        ko_d_model=d_model,
                        en_d_model=d_model,
                        max_seq_len=max_length,
                        head_num=head_num,
                        dropout=dropout,
                        N=N)

    dataset = TranslationDataset(tokenizer1=tokenizer1, tokenizer2=tokenizer2, file_path=data_path,
                                 max_length=max_length)

    if os.path.isfile(f'{checkpoint_path}/{model_name}.pth'):
        checkpoint = torch.load(f'{checkpoint_path}/{model_name}.pth', map_location=device)
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        global_steps = checkpoint['train_step']

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'{checkpoint_path}/{model_name}-.pth loaded')
        model.eval()
    pre = []
    tar = []
    for j in range(len(dataset)):

        encoder_input = dataset[j]['input']
        encoder_mask =  dataset[j]['input_mask']
        input_str = dataset[j]['input_str']
        pp = dataset[j]['target']
        target = torch.ones(1, 1).fill_(tokenizer1.cls_token_id).type_as(encoder_input)
        encoder_output = model.encode(encoder_input.unsqueeze(0), encoder_mask)
        for i in range(max_length - 1):
            lm_logits = model.decode(encoder_output, encoder_mask, target,
                                     Variable(subsequent_mask(target.size(1)).type_as(encoder_input.data)))
            prob = lm_logits[:, -1]
            _, next_word = torch.max(prob, dim=1)


            if next_word.data[0] == tokenizer1.pad_token_id or next_word.data[0] == tokenizer1.sep_token_id:
                print(j)
                # print(f'ko: {input_str} en: {tokenizer1.decode(target.squeeze().tolist(), skip_special_tokens=True)}')
                pre.append([tokenizer1.decode(target.squeeze().tolist(), skip_special_tokens=True).split(' ')])
                tar.append(tokenizer1.decode(pp, skip_special_tokens=True).split(' '))
                # print((
                #       tokenizer1.decode(target.squeeze().tolist(), skip_special_tokens=True).split(' '), (tokenizer1.decode(pp, skip_special_tokens=True)).split(' ')))
                # print(bleu.sentence_bleu(
                #       [tokenizer1.decode(target.squeeze().tolist(), skip_special_tokens=True).split(' ')], (tokenizer1.decode(pp, skip_special_tokens=True)).split(' '),weights=(0.25, 0.25, 0.25, 0.25)))
                break
            target = torch.cat((target[0], next_word))
            target = target.unsqueeze(0)
    a = bleu.corpus_bleu(pre, tar, weights=(1, 0, 0, 0) )
    print(a)