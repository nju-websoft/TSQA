import argparse
import os
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--nn', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
# parser.add_argument('--gpu', type=str, required=True)

args = parser.parse_args()

print(f'args:\n{str(args)}')


def contruct_mc_train_script(i):
    k = args.k
    model_path = 'outputs/c3_model'
    output_path = f'kbert_multi_choice_{k}_fold_{i}_{args.nn}'

    script = f'''python run_kbert_cls.py \\
        --do_eval \\
        --t_result_output_path results/t_result_{i}_{args.nn}.json \\
        --pretrained_model_path {model_path} \\
        --train_path datasets/qa/{args.data_path}/fold_{i}/train.json \\
        --dev_path datasets/qa/{args.data_path}/fold_{i}/dev.json \\
        --test_path datasets/qa/{args.data_path}/fold_{i}/test.json \\
        --vocab_path models/vocab.txt \\
        --epochs_num 5 --batch_size 16 \\
        --output_model_path outputs/{output_path} \\
        --tokenizer word \\
        --encoder bert \\
        --seq_length 256 \\
        --learning_rate 1e-5 \\
        --max_entities 2 \\
        --task_name multi_choice \\
        --labels_num 2  \\
        --kfold \\
        --fold {i} \\
        --result_output results/results_kbert_multi_choice_{k}_flod_{i}_{args.nn}.txt \\
        --mlc_test_eval_output mlc_epoch_15_result_no_entities.txt \\
        --kg_rank_file brain/kgs/sentences_ranking_kg.rank
    '''

    return script


class myThread(threading.Thread):
    def __init__(self, threadID, name, script, gpu):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.script = script
        self.gpu = gpu

    def run(self):
        print("begin: " + self.name)
        tmp_script = open(f'tmp_sh/{self.name}_as_sh.sh', 'w', encoding='utf-8')
        tmp_script.write(self.script)
        tmp_script.flush()
        tmp_script.close()
        os.system(f'CUDA_VISIBLE_DEVICES={self.gpu} bash tmp_sh/{self.name}_as_sh.sh')
        print("exit: " + self.name)


# visible_gpu = [int(gpu) for gpu in args.gpu]

k = args.k

flod_index = enumerate(range(k))
while True:
    threads = []
    try:
        used_gpu = []
        threads = []
        visible_gpu = [3]
        print(f'visible gpu: {str(visible_gpu)}')
        while len(used_gpu) < len(visible_gpu):
            _, i = next(flod_index)
            for gpu in visible_gpu:
                if not gpu in used_gpu:
                    current_gpu = gpu
                    break
            used_gpu.append(current_gpu)
            script = contruct_mc_train_script(i)
            threads.append(myThread(1, f'run_kbert_multi_choice_{k}_flod_{i}_{args.nn}_test', script, current_gpu))
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
