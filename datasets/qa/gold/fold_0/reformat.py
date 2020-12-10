import json

for name in ['train', 'test', 'dev']:
    data = json.load(open(f'{name}.json', 'r', encoding='utf-8'))
    output = open(f'{name}.txt', 'w', encoding='utf-8')
    output.write('\t'.join(
        ['question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer'])+'\n')
    for question in data:
        scenario = question[0][1]+question[0][0]
        for sub_question in question[1]:
            question_text = sub_question['question']
            choice = sub_question['choice']
            if '①' in ''.join(choice) or 'a' in ''.join(choice) or 'A' in ''.join(choice):
                continue
            answer = chr(ord('A')+choice.index(sub_question['answer']))
            line = f'{scenario}{question_text}\t' + \
                '\t'.join(choice)+'\t'+answer+'\n'
            output.write(line)
            output.flush()
    output.flush()
