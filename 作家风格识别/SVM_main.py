import jieba as jb
import numpy as np
import pickle


# # 5 个分类器分类
with open('results/svm_model.pkl', 'rb') as f:
    int2author, word2int, svm_lst = pickle.load(f)


def predict(text):
    word_num = len(word2int)
    feature = np.zeros(word_num)
    for word in jb.lcut(text):
        if word in word2int:
            feature[word2int[word]] += 1
    feature /= feature.sum()
    probabilities = []
    for svm_i in svm_lst:
        pred_y = svm_i.predict_proba([feature])[0]
        probabilities.append(pred_y[1])

    author_idx = max(enumerate(probabilities), key=lambda x: x[1])[0]
    return int2author[author_idx]