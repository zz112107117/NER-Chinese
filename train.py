import torch
import torch.optim as optim

from utils import f1_score
from data import MyDataset
from model import BiLSTMCRF

def train():
    # 定义训练集
    train_dataset = MyDataset(batch_size = 20, tags = ["ORG", "PER"])
    # 定义测试集
    test_dataset = MyDataset(batch_size = 20, data_type = "test", word2id = train_dataset.word2id, tag2id = train_dataset.tag2id)
    test_batch = test_dataset.iteration()

    # 定义模型
    model = BiLSTMCRF(tag2id = train_dataset.tag2id,
                      word2id_size = len(train_dataset.word2id),
                      batch_size = 20,
                      embedding_dim = 100,
                      hidden_dim = 128)

    # 定义优化器
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        index = 0
        for batch in train_dataset.get_batch():
            index += 1
            model.zero_grad()

            # 读取一个batch的数据并转换为tensor
            batch_sentences, batch_tags, batch_len = zip(*batch)
            batch_sentences_tensor = torch.tensor(batch_sentences, dtype = torch.long)
            batch_tags_tensor = torch.tensor(batch_tags, dtype = torch.long)
            batch_len_tensor = torch.tensor(batch_len, dtype = torch.long)

            loss = model.neg_log_likelihood(batch_sentences_tensor,
                                            batch_tags_tensor,
                                            batch_len_tensor)
            print("""epoch [{}] \tloss {:.2f}""".format(epoch, loss.tolist()[0]))
            # 测试集性能
            print("\teval")
            batch_sentences, batch_tags, batch_len = zip(*test_batch.__next__())
            _, batch_paths = model(batch_sentences)
            for tag in ["ORG", "PER"]:
                f1_score(batch_tags, batch_paths, tag, train_dataset.tag2id)

            # 反向传播+优化
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()