import sys

import torch

from data import MyDataset
from model import BiLSTMCRF
from train import train
from predict import predict

if __name__ == "__main__":
    # 特判
    if len(sys.argv) < 2:
        print("menu:\n\ttrain\n\tpredict")
        exit()

    # 定义训练集
    train_dataset = MyDataset(batch_size = 32, tags = ["ORG", "PER"])
    # 定义测试集
    word2id, tag2id = train_dataset.word2id, train_dataset.tag2id
    test_dataset = MyDataset(batch_size = 32, data_type = "test", word2id = word2id, tag2id = tag2id)

    if sys.argv[1] == "train":
        # 定义模型
        model = BiLSTMCRF(tag2id = tag2id,
                          word2id_size = len(word2id),
                          batch_size = 32,
                          embedding_dim = 100,
                          hidden_dim = 128)
        # 训练模型
        train(train_dataset = train_dataset,
              test_dataset = test_dataset,
              model = model,
              tag2id = tag2id)

    elif sys.argv[1] == "predict":
        # 定义模型
        model = BiLSTMCRF(tag2id = tag2id,
                          word2id_size = len(word2id),
                          batch_size = 1,
                          embedding_dim = 100,
                          hidden_dim = 128)
        # 加载模型参数
        model.load_state_dict(torch.load("models/params.pkl"))
        print("model restore success!")
        # 预测
        predict(model = model,
                word2id= word2id,
                tag2id = tag2id)