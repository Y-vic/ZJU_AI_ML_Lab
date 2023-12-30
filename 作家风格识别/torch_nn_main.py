import torch
import torch.nn as nn
import jieba as jb


config_path = 'results/my_model.pth'
config = torch.load(config_path)

word2int = config['word2int']
int2author = config['int2author']
word_num = len(word2int)
model = nn.Sequential(
    nn.Linear(word_num, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 6),
)
model.load_state_dict(config['model'])
int2author.append(int2author[0])


def predict(text):
    feature = torch.zeros(word_num)
    for word in jb.lcut(text):
        if word in word2int:
            feature[word2int[word]] += 1
    feature /= feature.sum()
    model.eval()
    out = model(feature.unsqueeze(dim=0))
    pred = torch.argmax(out, 1)[0]
    return int2author[pred]


