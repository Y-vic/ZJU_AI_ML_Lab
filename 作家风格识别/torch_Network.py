import os
import numpy as np
import jieba as jb
import jieba.analyse
import torch
import torch.nn as nn
from torch.utils import data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

int2author = ['LX', 'MY', 'QZS', 'WXB', 'ZAL']
author_num = len(int2author)
author2int = {author: i for i, author in enumerate(int2author)}


# dataset = {(sentence, label), }
dataset_init = []
path = 'dataset/'
for file in os.listdir(path):
    if not os.path.isdir(file) and not file[0] == '.':  # 跳过隐藏文件和文件夹
        with open(os.path.join(path, file), 'r',  encoding='UTF-8') as f:  # 打开文件
            for line in f.readlines():
                dataset_init.append((line, author2int[file[:-4]]))


# 将片段组合在一起后进行词频统计
str_full = ['' for _ in range(author_num)]
for sentence, label in dataset_init:
    str_full[label] += sentence

# 词频特征统计，取出各个作家前 300 的词
words = set()
for label, text in enumerate(str_full):
    for word in jb.analyse.extract_tags(text, topK=300, withWeight=False):
        words.add(word)

int2word = list(words)
word_num = len(int2word)
word2int = {word: i for i, word in enumerate(int2word)}

features = torch.zeros((len(dataset_init), word_num))
labels = torch.zeros(len(dataset_init))
for i, (sentence, author_idx) in enumerate(dataset_init):
    feature = torch.zeros(word_num, dtype=torch.float)
    for word in jb.lcut(sentence):
        if word in words:
            feature[word2int[word]] += 1
    if feature.sum():
        feature /= feature.sum()
        features[i] = feature
        labels[i] = author_idx
    else:
        labels[i] = 5  # 表示识别不了作者

dataset = data.TensorDataset(features, labels)

# 划分数据集
valid_split = 0.3
train_size = int((1 - valid_split) * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
# 创建一个 DataLoader 对象
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = data.DataLoader(test_dataset, batch_size=1000, shuffle=True)


model = nn.Sequential(
    nn.Linear(word_num, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 6),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_acc = 0
best_model = model.cpu().state_dict().copy()

for epoch in range(40):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        out = model(b_x)
        loss = loss_fn(out, b_y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = np.mean((torch.argmax(out, 1) == b_y).cpu().numpy())

        with torch.no_grad():
            for b_x, b_y in valid_loader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                out = model(b_x)
                valid_acc = np.mean((torch.argmax(out, 1) == b_y).cpu().numpy())
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model.cpu().state_dict().copy()
    print('epoch:%d | valid_acc:%.4f' % (epoch, valid_acc))

print('best accuracy:%.4f' % (best_acc, ))
torch.save({
    'word2int': word2int,
    'int2author': int2author,
    'model': best_model,
}, 'results/nn_model.pth')

