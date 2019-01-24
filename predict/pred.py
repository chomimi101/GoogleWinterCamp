# coding: utf-8

import json
import torch
from predict.utils import vocab_path, pad, labels_vocab, Args, post_cleaner, SEP


def pred(args, sentence, vocab, model):
    inputs = [[vocab.get(word, 0) for word in sentence]]
    inputs_len = torch.LongTensor([min(len(inputs), args.max_len)])
    # print(inputs[0][:10])
    inputs = torch.LongTensor(pad(inputs, max_len=args.max_len))
    logits = model((inputs, inputs_len))
    pred = torch.max(logits, -1)[1]
    label_index = pred.data.cpu().numpy().tolist()[0]
    for mbti, index in labels_vocab.items():
        if index == label_index:
            return mbti


def pred_main(sentence):
    args = Args()
    with open(vocab_path) as f:
        for line in f:
            vocab = json.loads(line.strip())
    sentence = post_cleaner(sentence)
    sentence = sentence.split(' ')
    model = torch.load('./res/lstm_torch.model')
    label = pred(args, sentence, vocab, model)
    return label


if __name__ == '__main__':
    sentence = ''
    with open('../resources/tweets/HC.txt', encoding='utf-8') as f:
        for line in f:
            sentence += line.strip()
    sentence = "I like to think, often find myself immersed in my own world, but I also like to communicate with people, I want to know different people, I can grow from them. At the same time, I am afraid of too much interaction with people, because I have more things I want to do, I will pay more attention to other people's opinions, although I know that this is not right. I am more shy to strangers, but can be very humorous to acquaintances. I hope that I can do something useful to society."
    sentence = "I am so happy."
    print(pred_main(sentence))

    # info = {}
    # info['data'] = []
    # info['data'].append({'name': 'Warren Buffet', 'type': 'ISTJ', 'type_description': '物流师：实际且注重事实的个人，可靠性不容怀疑', 'description': ''})
    # info['data'].append({'name': 'Mother Theresa', 'type': 'ISFJ', 'type_description': '守卫者：非常专注而温暖的守护者，时刻准备着保护爱着的人们', 'description': ''})
    # info['data'].append({'name': 'Mahatma Gandhi', 'type': 'INFJ', 'type_description': '提倡者：安静而神秘，同时鼓舞人心且不知疲倦的理想主义者', 'description': ''})
    # info['data'].append({'name': 'Muck Zuckerberg', 'type': 'INTJ', 'type_description': '建筑师：富有想象力和战略性的思想家，一切皆在计划之中', 'description': ''})
    # info['data'].append({'name': 'Steve Jobs', 'type': 'ISTP', 'type_description': '鉴赏家：大胆而实际的实验家，擅长使用任何形式的工具', 'description': ''})
    # info['data'].append({'name': 'Michael Jackson', 'type': 'ISFP', 'type_description': '探险家：灵活有魅力的艺术家，时刻准备着探索和体验新鲜事物', 'description': ''})
    # info['data'].append({'name': 'J K Rowling', 'type': 'INFP', 'type_description': '调停者：诗意，善良的利他主义者，总是热情地为正当理由提供帮助', 'description': ''})
    # info['data'].append({'name': 'Jimmy Wales', 'type': 'INTP', 'type_description': '逻辑学家：具有创造力的发明家，对知识有着止不住的渴望', 'description': ''})
    # info['data'].append({'name': 'Donald Trump', 'type': 'ESTP', 'type_description': '企业家：聪明，精力充沛善于感知的人们，真心享受生活在边缘', 'description': ''})
    # info['data'].append({'name': 'Larry Ellison', 'type': 'ESFP', 'type_description': '表演者：自发的，精力充沛而热情的表演者，生活在他们周围永不无聊', 'description': ''})
    # info['data'].append({'name': 'Walt Disney', 'type': 'ENFP', 'type_description': '竞选者：热情，有创造力爱社交的自由自在的人，总能找到理由微笑', 'description': ''})
    # info['data'].append({'name': 'Barack Obama', 'type': 'ENTP', 'type_description': '辩论家：聪明好奇的思想者，不会放弃任何智力上的挑战', 'description': ''})
    # info['data'].append({'name': 'Steve Ballmer', 'type': 'ESTJ', 'type_description': '总经理：出色的管理者，在管理事情或人的方面无与伦比', 'description': ''})
    # info['data'].append({'name': 'Sam Walton', 'type': 'ESFJ', 'type_description': '执政官：极有同情心，爱交往受欢迎的人们，总是热心提供帮助', 'description': ''})
    # info['data'].append({'name': 'Oprah Winfrey', 'type': 'ENFJ', 'type_description': '主人公：富有魅力鼓舞人心的领导者，有使听众着迷的能力', 'description': ''})
    # info['data'].append({'name': 'Bill Gates', 'type': 'ENTJ', 'type_description': '指挥官：大胆，富有想象力且意志强大的领导者，总能找到或创造解决办法', 'description': ''})
    # with open('doc.txt', 'w') as f:
    #     f.write(json.dumps(info))

    # info = {}
    # info['data'] = []
    # info['data'].append({'name': 'Muck Zuckerberg', 'type': 'INTJ', 'type_description': '“Architect”: Imaginative and strategic thinkers, with a plan for everything.', 'description': ''})
    # info['data'].append({'name': 'Jimmy Wales', 'type': 'INTP', 'type_description': '逻辑学家：具有创造力的发明家，对知识有着止不住的渴望', 'description': ''})
    # info['data'].append({'name': 'Bill Gates', 'type': 'ENTJ', 'type_description': '指挥官：大胆，富有想象力且意志强大的领导者，总能找到或创造解决办法', 'description': ''})
    # info['data'].append({'name': 'Barack Obama', 'type': 'ENTP', 'type_description': '辩论家：聪明好奇的思想者，不会放弃任何智力上的挑战', 'description': ''})
    #
    # info['data'].append({'name': 'Mahatma Gandhi', 'type': 'INFJ', 'type_description': '“Advocate”: Quiet and mystical, yet very inspiring and tireless idealists.', 'description': ''})
    # info['data'].append({'name': 'J K Rowling', 'type': 'INFP', 'type_description': '调停者：诗意，善良的利他主义者，总是热情地为正当理由提供帮助', 'description': ''})
    # info['data'].append({'name': 'Oprah Winfrey', 'type': 'ENFJ', 'type_description': '主人公：富有魅力鼓舞人心的领导者，有使听众着迷的能力', 'description': ''})
    # info['data'].append({'name': 'Walt Disney', 'type': 'ENFP', 'type_description': '竞选者：热情，有创造力爱社交的自由自在的人，总能找到理由微笑', 'description': ''})
    #
    # info['data'].append({'name': 'Warren Buffet', 'type': 'ISTJ', 'type_description': '物流师：实际且注重事实的个人，可靠性不容怀疑', 'description': ''})
    # info['data'].append({'name': 'Mother Theresa', 'type': 'ISFJ', 'type_description': '守卫者：非常专注而温暖的守护者，时刻准备着保护爱着的人们', 'description': ''})
    # info['data'].append({'name': 'Steve Ballmer', 'type': 'ESTJ', 'type_description': '总经理：出色的管理者，在管理事情或人的方面无与伦比', 'description': ''})
    # info['data'].append({'name': 'Sam Walton', 'type': 'ESFJ', 'type_description': '执政官：极有同情心，爱交往受欢迎的人们，总是热心提供帮助', 'description': ''})
    #
    # info['data'].append({'name': 'Steve Jobs', 'type': 'ISTP', 'type_description': '鉴赏家：大胆而实际的实验家，擅长使用任何形式的工具', 'description': ''})
    # info['data'].append({'name': 'Michael Jackson', 'type': 'ISFP', 'type_description': '探险家：灵活有魅力的艺术家，时刻准备着探索和体验新鲜事物', 'description': ''})
    # info['data'].append({'name': 'Donald Trump', 'type': 'ESTP', 'type_description': '企业家：聪明，精力充沛善于感知的人们，真心享受生活在边缘', 'description': ''})
    # info['data'].append({'name': 'Larry Ellison', 'type': 'ESFP', 'type_description': '表演者：自发的，精力充沛而热情的表演者，生活在他们周围永不无聊', 'description': ''})
    # with open('doc.txt', 'w') as f:
    #     f.write(json.dumps(info))
