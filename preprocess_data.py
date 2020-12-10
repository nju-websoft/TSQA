import argparse
import os
import threading

from sklearn.model_selection import KFold

if not os.path.exists('tmp'):
    os.makedirs('tmp')
if not os.path.exists('tmp_sh'):
    os.makedirs('tmp_sh')
if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('results'):
    os.makedirs('results')

parser = argparse.ArgumentParser()
parser.add_argument('--all_data_path', type=str, required=True)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--k', type=int, default=5)
args = parser.parse_args()


class myThread(threading.Thread):
    def __init__(self, threadID, name, script, gpu):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.script = script
        self.gpu = gpu

    def set_gpu(self, gpu):
        self.gpu = gpu

    def run(self):
        print("begin: " + self.name)
        tmp_script = open(f'tmp_sh/{self.name}_sh.sh', 'w', encoding='utf-8')
        tmp_script.write(self.script)
        tmp_script.flush()
        tmp_script.close()
        os.system(f'CUDA_VISIBLE_DEVICES={self.gpu} bash tmp_sh/{self.name}_sh.sh')
        print("exit: " + self.name)


datas = open(args.all_data_path, 'r', encoding='utf-8').read().split('\n')[1:-1]
guid_datas_map = {}
for data in datas:
    guid = data.split('\t')[3].split('-')[0]
    if not guid in guid_datas_map:
        guid_datas_map[guid] = []
    guid_datas_map[guid].append(data)


def data_2_file(data, data_path, data_file):
    if not os.path.exists('/'.join(data_path.split('/')[:-1])):
        os.mkdir('/'.join(data_path.split('/')[:-1]))
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    output = open(os.path.join(data_path, data_file), 'w', encoding='utf-8')
    output.write('text_a\ttext_b\tlabel\tguid\ttemplate\tmost-value\tin-decrease\tspeci\tcmp\tmore-less\ttrend\n')
    for d in data:
        output.write('\n'.join(guid_datas_map[d]) + '\n')
    output.flush()
    output.close()


k = args.k
D = [f'guid_{i}' for i in range(556)]

kf = KFold(k, shuffle=True, random_state=args.random_seed)
for i, (train_indexes, test_indexes) in enumerate(kf.split(D)):
    train_ids = [D[index] for index in train_indexes]
    train_ids_len = len(train_ids)
    train_ids, dev_ids = train_ids[:int(train_ids_len * 0.8)], train_ids[int(train_ids_len * 0.8):]
    test_ids = [D[index] for index in test_indexes]
    print(f'data to file flod {i}')
    data_2_file(train_ids, f'datasets/sentences_ranking/cv_{k}_flod_{i}/as', 'train.txt')
    data_2_file(dev_ids, f'datasets/sentences_ranking/cv_{k}_flod_{i}/as', 'dev.txt')
    data_2_file(test_ids, f'datasets/sentences_ranking/cv_{k}_flod_{i}/as', 'test.txt')
