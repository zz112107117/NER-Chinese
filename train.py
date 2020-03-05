import torch
import torch.optim as optim
import time

from utils import f1_score

def train_model(train_dataset, test_dataset, model, tag2id):
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

            # 保存模型参数
            torch.save(model.state_dict(), 'models/params.pkl')

        # 训练集loss（每个batch）
        print("epoch: {}\tloss: {:.2f}\ttime: {:.1f} sec".format(epoch + 1, total_loss / batch_count, time.time() - start))
        # 测试集性能
        print("\t** eval **")
        f1_score(test_dataset, model, tag2id)