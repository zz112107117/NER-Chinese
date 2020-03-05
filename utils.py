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
            #print('!!!')
            begin = index
        # 记录实体的终点
        elif each_tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            #print('???')
            end = index    # 实体的终点
            positions.append([begin, end])
        # 其他
        elif tag == o_tag:
            #print("...")
            begin = -1
        last_tag = each_tag    # 当前时刻的上一个each_tag
    return positions

# 计算p，r，f1
def f1_score(batch_tags, batch_paths, tag, tag2id):
    origin = 0.
    found = 0.
    right = 0.

    for each in zip(batch_tags, batch_paths):
        # tags是标准值，path是预测值
        tags, path = each
        tar_positions = get_entity_position(tags, tag, tag2id)
        path_positions = get_entity_position(path, tag, tag2id)
        #print(tar_positions, path_positions)
        origin += len(tar_positions)    # 标准值中实体的个数
        found += len(path_positions)    # 预测值中实体的个数

        # 预测正确的个数
        for position in path_positions:
            if position in tar_positions:
                right += 1

    precision = 0. if found == 0. else (right / found)
    recall = 0. if origin == 0. else (right / origin)
    f1 = 0. if precision + recall == 0. else (2 * precision * recall) / (precision + recall)
    #print(origin, found, right)
    print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(tag, recall, precision, f1))