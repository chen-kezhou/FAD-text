import torch
from torch.utils.data import DataLoader, Dataset
import json
from transformers import *
f_train = open('/data1/kezhou/homework/src/dataset/train.json','r')
content_train = f_train.read()
f_dev = open('/data1/kezhou/homework/src/dataset/dev.json','r')
content_dev = f_dev.read()
bert_path = '/data1/kezhou/pretrained_large_model/bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

class FDAdata(Dataset):  # Dataset
    def __init__(self,content):
        self.data = json.loads(content)  # list
    def __getitem__(self, idx):  
        name = self.data[idx]['user']['name']
        screen_name = self.data[idx]['user']['screen_name']
        description = self.data[idx]['user']['description']
        # encoded_bert_sent = bert_tokenizer(description)
        label = self.data[idx]['label']
        label = 0 if label=='human' else 1  # 0代表human, 1代表bot
        return description, label
    def __len__(self):
        return len(self.data)

def collate_fn(batch_data):
    label = []
    bert_details = []
    for i, sample in enumerate(batch_data):
        description,data_label=sample
        text = " ".join(description)
        encoded_bert_sent = bert_tokenizer.encode_plus(
                    text, max_length=50, add_special_tokens=True, truncation=True, padding='max_length', pad_to_max_length=True)
        bert_details.append(encoded_bert_sent)
        label.append(data_label)
    bert_sentences = torch.LongTensor([sample['input_ids'] for sample in bert_details])
    bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
    bert_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
    label = torch.Tensor(label).reshape(-1,1)

    # encoded_text = bert_tokenizer.encode(text=batch_text, padding=True, return_tensors="pt")
    return bert_sentences, bert_sentence_types, bert_att_mask, label 

def getloader(mode,batch_size,shuffle):  # mode决定数据集类型. 本函数返回dataloader.
    content = content_train if mode == 'train' else content_dev
    dataset = FDAdata(content)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)
    return data_loader

# train_loader = getloader(mode='train',batch_size=3,shuffle=False)  # dataloader 返回内容如下所示
# for i, data in enumerate(train_loader):
#     if i ==0:
#         bert_sentences, bert_sentence_types, bert_att_mask, label = data
#         print(type(bert_sentences))  # tensor
#         print(bert_sentences)
#         print(type(label))  # tensor
#         print(label.shape)  # batch_size, 1
#         print(label)  # tensor([1., 1., 1.])