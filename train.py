import torch
import torch.optim as optim
import time

from utils import f1_score
from data import MyDataset
from model import BiLSTMCRF

def train():
    # 定义训练集
    train_dataset = MyDataset(batch_size = 32, tags = ["ORG", "PER"])
    # 定义测试集
    word2id, tag2id = train_dataset.word2id, train_dataset.tag2id
    test_dataset = MyDataset(batch_size = 32, data_type = "test", word2id = word2id, tag2id = tag2id)

    # 定义模型
    model = BiLSTMCRF(tag2id = tag2id,
                      word2id_size = len(word2id),
                      batch_size = 32,
                      embedding_dim = 100,
                      hidden_dim = 128)

    # 定义优化器
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        total_loss = 0.    # 一轮训练的loss
        batch_count = 0.    # batch个数
        start = time.time()    # 计时

        for batch in train_dataset.get_batch():
            model.zero_grad()

            # 读取一个batch的数据并转换为tensor
            batch_sentences, batch_tags, batch_len = zip(*batch)
            batch_sentences_tensor = torch.tensor(batch_sentences, dtype = torch.long)
            batch_tags_tensor = torch.tensor(batch_tags, dtype = torch.long)
            batch_len_tensor = torch.tensor(batch_len, dtype = torch.long)

            loss = model.neg_log_likelihood(batch_sentences_tensor,
                                            batch_tags_tensor,
                                            batch_len_tensor)

            total_loss += loss.tolist()[0]
            batch_count += 1

            # 反向传播+优化
            loss.backward()
            optimizer.step()

        # 训练集loss（每个batch）
        print("epoch: {}\tloss: {:.2f}\ttime: {:.1f} sec".format(epoch + 1, total_loss / batch_count, time.time() - start))
        # 测试集性能
        print("\t** eval **")
        f1_score(test_dataset, model, tag2id)

if __name__ == "__main__":
    train()