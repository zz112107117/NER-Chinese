# 找出所有实体
def get_entity_position(path, tag, tag2id):
    begin_tag = tag2id.get("B-" + tag)
    mid_tag = tag2id.get("I-" + tag)
    end_tag = tag2id.get("E-" + tag)
    o_tag = tag2id.get("O")

    begin = -1    # 实体的起点
    positions = []    # 所有实体的起点和终点
    last_tag = 0

    for index, each_tag in enumerate(path):
        # 记录实体的起点
        if each_tag == begin_tag:
            begin = index
        # 记录实体的终点
        elif each_tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index    # 实体的终点
            positions.append([begin, end])
        # 其他
        elif tag == o_tag:
            begin = -1
        last_tag = each_tag    # 当前时刻的上一个each_tag
    return positions

# 统计origin，found，right
def count_ofr(batch_tags, batch_paths, tag, tag2id):
    origin = 0.
    found = 0.
    right = 0.

    for each in zip(batch_tags, batch_paths):
        # tags是标准值，path是预测值
        tags, path = each
        tar_positions = get_entity_position(tags, tag, tag2id)
        path_positions = get_entity_position(path, tag, tag2id)
        origin += len(tar_positions)    # 标准值中实体的个数
        found += len(path_positions)    # 预测值中实体的个数

        # 预测正确的个数
        for position in path_positions:
            if position in tar_positions:
                right += 1

    return origin, found, right

# 计算f1值
def f1_score(test_dataset, model, tag2id):
    ORG_o, ORG_f, ORG_r = 0., 0., 0.
    PER_o, PER_f, PER_r = 0., 0., 0.

    # 每个标签
    for tag in ["ORG", "PER"]:
        # 每个batch
        for batch in test_dataset.get_batch():
            batch_sentences, batch_tags, batch_len = zip(*batch)
            _, batch_paths = model(batch_sentences)

            o, f, r = count_ofr(batch_tags, batch_paths, tag, tag2id)
            if tag == "ORG":
                ORG_o += o
                ORG_f += f
                ORG_r += r
            elif tag == "PER":
                PER_o += o
                PER_f += f
                PER_r += r

        if tag == "ORG":
            precision = 0. if ORG_f == 0. else (ORG_r / ORG_f)
            recall = 0. if ORG_o == 0. else (ORG_r / ORG_o)
            f1 = 0. if precision + recall == 0. else (2 * precision * recall) / (precision + recall)
            print("\t{}:\tprecision: {:.1f},\trecall: {:.1f},\tf1: {:.1f}".format(tag, precision * 100, recall * 100, f1 * 100))
        if tag == "PER":
            precision = 0. if PER_f == 0. else (PER_r / PER_f)
            recall = 0. if PER_o == 0. else (PER_r / PER_o)
            f1 = 0. if precision + recall == 0. else (2 * precision * recall) / (precision + recall)
            print("\t{}:\tprecision: {:.1f},\trecall: {:.1f},\tf1: {:.1f}".format(tag, precision * 100, recall * 100, f1 * 100))

    o = ORG_o + PER_o
    f = ORG_f + PER_f
    r = ORG_r + PER_r
    precision = 0. if f == 0. else (r / f)
    recall = 0. if o == 0. else (r / o)
    f1 = 0. if precision + recall == 0. else (2 * precision * recall) / (precision + recall)
    print("\toverall:\tprecision: {:.1f},\trecall: {:.1f},\tf1: {:.1f}\n".format(precision * 100, recall * 100, f1 * 100))