import torch

from utils import format_result, get_entity_position

def predict(model, word2id, tag2id):
    input_str = input("请输入文本: ")
    input_vec = [word2id.get(i, 0) for i in input_str]
    sentences = torch.tensor(input_vec).view(1, -1)
    _, paths = model(sentences)

    entities = []
    for tag in ["ORG", "PER"]:
        tags = get_entity_position(paths[0], tag, tag2id)
        entities += format_result(tags, input_str, tag)
    print(entities)