import torch

from utils import format_result, get_entity_position

def predict_model(model, word2id, tag2id):
    input_str = input("请输入文本: ")
    # 将输入的每个字转换为对应的id
    input2id = [word2id.get(i, 0) for i in input_str]
    # 转换为tensor（修改shape）
    sentences = torch.tensor(input2id).view(1, -1)
    # 转换为list
    sentences = sentences.tolist()

    _, paths = model(sentences)

    entities = []
    for tag in ["ORG", "PER"]:
        positions = get_entity_position(paths[0], tag, tag2id)
        entities += format_result(positions, input_str, tag)

    # 输出结果
    print(entities)