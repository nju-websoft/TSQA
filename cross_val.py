import argparse
import os
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--nn', type=str)
parser.add_argument('--gpu', type=str, required=True)

args = parser.parse_args()

print(f'args:\n{str(args)}')


def construct_train_script(log_file, data_path, i=-1):
    model_path = 'models/uer_model.bin'
    output_path = f'kbert_{k}_flod_{i}_{args.nn}'
    script = f'''python run_kbert_cls.py \\
    --pretrained_model_path {model_path} \\
    --vocab_path /home/xli/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt \\
    --train_path {data_path}/train.txt \\
    --dev_path {data_path}/dev.txt \\
    --test_path {data_path}/test.txt \\
    --epochs_num 5 --batch_size 16 \\
    --output_model_path outputs/{output_path} \\
    --tokenizer word \\
    --encoder bert \\
    --seq_length 256 \\
    --learning_rate 1e-5 \\
    --max_entities 0 \\
    --task_name as \\
    --labels_num 2  \\
    --kfold \\
    --fold {i} \\
    --result_output results/results_5_flod_{i}_{args.nn}_dev_test.txt \\
    --mlc_test_eval_output mlc_epoch_15_result.txt \\
    --kg_rank_file brain/kgs/sentences_ranking_kg.rank > {log_file}
'''
    return script


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
        tmp_script = open(f'tmp_sh/{self.name}_as_sh.sh', 'w', encoding='utf-8')
        tmp_script.write(self.script)
        tmp_script.flush()
        tmp_script.close()
        os.system(f'CUDA_VISIBLE_DEVICES={self.gpu} bash tmp_sh/{self.name}_as_sh.sh')
        print("exit: " + self.name)


visible_gpu = [int(gpu) for gpu in args.gpu.split(',')]

k = args.k

flod_index = enumerate(range(k))
while True:
    threads = []
    try:
        used_gpu = []
        threads = []
        print(f'visible gpu: {str(visible_gpu)}')
        while len(used_gpu) < len(visible_gpu):
            _, i = next(flod_index)
            for gpu in visible_gpu:
                if not gpu in used_gpu:
                    current_gpu = gpu
                    break
            used_gpu.append(current_gpu)
            script = construct_train_script(
                f'datasets/sentences_ranking/cv_{k}_flod_{i}/as/log_{k}_flod_{i}_{args.nn}.log',
                f'datasets/sentences_ranking/cv_{k}_flod_{i}/as', i=i)
            threads.append(myThread(1, f'run_{k}_flod_{i}_{args.nn}', script, current_gpu))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except StopIteration:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        break
