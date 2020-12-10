import json
import os


def read_dev_test_sentences(filename):
    sentences_map = {}
    datas = open(filename, 'r', encoding='utf-8').read().split('\n\n')
    for data in datas:
        data = data.strip().split('\n')
        guid = data[0]
        sentences_map[guid] = []
        for line in data[2:]:
            sentences_map[guid].append(line.split('\t')[1].strip())
    return sentences_map


def read_train_sentences(dev_test_guid_sentences_map):
    datas = open('datasets/qa/all_as_all.txt', 'r', encoding='utf-8').read().split('\n')[1:-1]
    guid_sentences_map = {}
    for data in datas:
        data = data.split('\t')
        guid = data[3]
        if guid in dev_test_guid_sentences_map:
            continue
        label = data[2]
        if label == '0':
            continue
        if guid not in guid_sentences_map:
            guid_sentences_map[guid] = []
        guid_sentences_map[guid].append(data[1])
    return guid_sentences_map


def get_train_dev_test_ids(fold):
    train_ids = []
    dev_ids = []
    test_ids = []
    for file_type in ['train', 'dev', 'test']:
        datas = open(f'datasets/sentences_ranking/cv_5_flod_{fold}/as/{file_type}.txt', 'r',
                     encoding='utf-8').read().split('\n')[1:-1]
        for data in datas:
            guid = data.split('\t')[3]
            if file_type == 'train':
                train_ids.append(guid)
            elif file_type == 'dev':
                dev_ids.append(guid)
            else:
                test_ids.append(guid)
    return list(set(train_ids)), list(set(dev_ids)), list(set(test_ids))


def instance_sort_by_id(instances):
    instances.sort(key=lambda x: int(x[-1].split('_')[1]))
    return instances


def get_GeoTSQA_data(guid_sentences_map, train_ids, dev_ids, test_ids, topk=1):
    datas = open('datasets/GeoTSQA.txt', 'r', encoding='utf-8').read().split('\n')[1:-1]
    train_instances = []
    test_instances = []
    dev_instances = []
    for data in datas:
        data = data.split('\t')
        guid = data[0]
        background = data[2]
        question = data[3]
        choices = data[4:8]
        answer = choices[ord(data[8]) - ord('A')]
        sentences = guid_sentences_map[guid][:topk]
        instance = [[background, ''.join(sentences)], [{'question': question, 'choice': choices, 'answer': answer}],
                    guid.split('-')[0]]
        if guid in train_ids:
            train_instances.append(instance)
        elif guid in dev_ids:
            dev_instances.append(instance)
        elif guid in test_ids:
            test_instances.append(instance)
        else:
            raise NotImplementedError
    return instance_sort_by_id(train_instances), instance_sort_by_id(dev_instances), instance_sort_by_id(test_instances)


def process(topk=1):
    if not os.path.exists(f'datasets/qa/TTGen_topk_{topk}'):
        os.mkdir(f'datasets/qa/TTGen_topk_{topk}')
    for fold in range(5):
        dev_test_sentences_map = read_dev_test_sentences(f'results/results_5_flod_{fold}_TTGen_dev_test.txt')
        train_sentences_map = read_train_sentences(dev_test_sentences_map)
        train_ids, dev_ids, test_ids = get_train_dev_test_ids(fold)
        train_instances, dev_instances, test_instances = get_GeoTSQA_data(
            {**train_sentences_map, **dev_test_sentences_map}, train_ids, dev_ids, test_ids, topk)
        if not os.path.exists(f'datasets/qa/TTGen_topk_{topk}/fold_{fold}'):
            os.mkdir(f'datasets/qa/TTGen_topk_{topk}/fold_{fold}')
        output_train = open(f'datasets/qa/TTGen_topk_{topk}/fold_{fold}/train.json', 'w', encoding='utf-8')
        json.dump(train_instances, output_train, ensure_ascii=False)
        output_train.close()
        output_dev = open(f'datasets/qa/TTGen_topk_{topk}/fold_{fold}/dev.json', 'w', encoding='utf-8')
        json.dump(dev_instances, output_dev, ensure_ascii=False)
        output_dev.close()
        output_test = open(f'datasets/qa/TTGen_topk_{topk}/fold_{fold}/test.json', 'w', encoding='utf-8')
        json.dump(test_instances, output_test, ensure_ascii=False)
        output_test.close()


def main():
    for k in range(6):
        process(k)


if __name__ == '__main__':
    main()
