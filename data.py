import copy

class MyDataset():
    def __init__(self, batch_size = 32, data_type = "train", tags = ["ORG", "PER"], word2id = None, tag2id = None):
        self.input_size = 0

        self.batch_size = batch_size
        self.data_type = data_type

        self.data = []    # [ [[sen1], [tag1]], [[sen2], [tag2]] ... ]
        self.batch_data = []    # [ batch1, batch2 ... ], batch1: [ [[sen1], [tag1]], [[sen2], [tag2]] ... ]

        self.word2id = {"unk": 0}
        self.tag2id = {"O": 0, "START": 1, "STOP": 2}

        self.generate_tags(tags)    # 所有标签

        # 训练集
        if data_type == "train":
            self.data_path = "data/train"
        # 测试集
        elif data_type == "test":
            self.data_path = "data/test"
            self.word2id = word2id
            self.tag2id = tag2id

        self.load_data()    # 把数据读入内存
        self.prepare_batch()

    # 生成所有标签
    def generate_tags(self, tags):
        self.tags = []
        for tag in tags:
            for prefix in ["B-", "I-", "E-"]:
                self.tags.append(prefix + tag)
        self.tags.append("O")

    # 把数据读入内存
    def load_data(self):
        sentence = []
        target = []

        with open(self.data_path) as f:
            for line in f:
                line = line[:-1]    # 去除"\n"

                if line == "end":
                    self.data.append([sentence, target])
                    # 清空
                    sentence = []
                    target = []
                    continue
                try:
                    word, tag = line.split(" ")
                except Exception:
                    continue
                # 更新word2id
                if self.data_type == "train" and word not in self.word2id:
                    self.word2id[word] = max(self.word2id.values()) + 1
                # 更新tag2id
                if self.data_type == "train" and tag not in self.tag2id:
                    self.tag2id[tag] = max(self.tag2id.values()) + 1

                # 把句子和标签转换成相应的id（默认值0针对测试集）
                sentence.append(self.word2id.get(word, 0))
                target.append(self.tag2id.get(tag, 0))

    # 将一个batch中的所有样本设为一致长度
    def pad_data(self, data):
        c_data = copy.deepcopy(data)
        max_length = max([len(i[0]) for i in c_data])    # 一个batch中，句子的最大长度
        for each in c_data:
            each.append(len(each[0]))    # each[2]，句子未填充的原长
            each[0] = each[0] + (max_length - len(each[0])) * [0]    # 填充句子
            each[1] = each[1] + (max_length - len(each[1])) * [0]    # 填充标签
        return c_data

    # 将数据集划分为batch
    def prepare_batch(self):
        index = 0
        while True:
            # 最后一个batch的情况
            if index + self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index: index + self.batch_size])
                self.batch_data.append(pad_data)
                index += self.batch_size    # 更新索引

    # 获得每个batch的数据
    def get_batch(self):
        for data in self.batch_data:
            yield data