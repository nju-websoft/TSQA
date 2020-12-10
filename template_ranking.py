import argparse
import os
import threading

from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--all_data_path', type=str, required=True)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--gpu', type=int, required=True)
args = parser.parse_args()


def contruct_mlc_train_script(log_file, data_path):
    script = f'''
    python run_kbert_cls.py \\
    --pretrained_model_path models/uer_model.bin \\
    --vocab_path models/vocab.txt \\
    --train_path {data_path}/train.txt \\
    --dev_path {data_path}/dev.txt \\
    --test_path {data_path}/dev_test.txt \\
    --epochs_num 15 --batch_size 16 \\
    --output_model_path outputs/kbert_mlc \\
    --tokenizer word \\
    --encoder bert \\
    --seq_length 256 \\
    --learning_rate 1e-5 \\
    --task_name mlc \\
    --labels_num 6  \\
    --kg_rank_file brain/kgs/template_rank_kg.rank \\
    --kfold
'''

    return script


class myThread(threading.Thread):
    def __init__(self, threadID, name, script):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.script = script

    def run(self):
        print("开始线程：" + self.name)
        tmp_script = open(f'tmp_sh/{self.name}_sh.sh', 'w', encoding='utf-8')
        tmp_script.write(self.script)
        tmp_script.flush()
        tmp_script.close()
        os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} bash tmp_sh/{self.name}_sh.sh')
        print("退出线程：" + self.name)


datas = open(args.all_data_path, 'r', encoding='utf-8').read().split('\n')[1:-1]
D = []  # 存放所有题目的id
guid_datas_map = {}
for data in datas:
    guid = data.split('\t')[0].split('-')[0]
    if not guid in guid_datas_map:
        guid_datas_map[guid] = []
    guid_datas_map[guid].append(data)
    if not guid in D:
        D.append(guid)


def data_2_file(data, data_path, data_file):
    if not os.path.exists('/'.join(data_path.split('/')[:-1])):
        os.mkdir('/'.join(data_path.split('/')[:-1]))
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    output = open(os.path.join(data_path, data_file), 'w', encoding='utf-8')
    output.write('guid\ttext_a\tmost-value\tin-decrease\tspeci\tcmp\tmore-less\ttrend\n')
    for d in data:
        output.write('\n'.join(guid_datas_map[d]) + '\n')
    output.flush()
    output.close()


print('question num: ' + str(len(D)))

k = args.k

D = [f'guid_{i}' for i in range(556)]
kf = KFold(k, shuffle=True, random_state=args.random_seed)
for i, (train_indexes, test_indexes) in enumerate(kf.split(D)):
    if i == 0:
        continue
    train_ids = [D[index] for index in train_indexes]
    train_ids_len = len(train_ids)
    train_ids, dev_ids = train_ids[:int(train_ids_len * 0.8)], train_ids[int(train_ids_len * 0.8):]
    test_ids = [D[index] for index in test_indexes]
    data_2_file(train_ids, f'datasets/sentences_ranking/cv_{k}_flod_{i}/mlc', 'train.txt')
    data_2_file(dev_ids, f'datasets/sentences_ranking/cv_{k}_flod_{i}/mlc', 'dev.txt')
    data_2_file(test_ids, f'datasets/sentences_ranking/cv_{k}_flod_{i}/mlc', 'test.txt')
    data_2_file(dev_ids + test_ids, f'datasets/sentences_ranking/cv_{k}_flod_{i}/mlc', 'dev_test.txt')
    mlc_script = contruct_mlc_train_script(f'datasets/sentences_ranking/cv_{k}_flod_{i}/mlc/data_preprocess_dev.log',
                                           f'datasets/sentences_ranking/cv_{k}_flod_{i}/mlc')
    thread = myThread(1, f'mlc_flod_{i}', mlc_script)
    thread.start()
    thread.join()
