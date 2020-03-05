# NER-Chinese
LSTM + CRF for Chinese NER


###step 1: need

    PyTorch 1.2.0
    
###step 2: train

    python3 main.py train_model
   
###step 3: predict

    python3 main.py predict_model
    
    model restore success!
    请输入文本: 我、张三和李四三个人一起去学习编程
    [{'begin': 2, 'end': 3, 'content': '张三', 'type': 'PER'}, {'begin': 5, 'end': 6, 'content': '李四', 'type': 'PER'}]
