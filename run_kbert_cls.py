# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss, precision_score, recall_score
from tqdm import tqdm

from brain import KnowledgeGraph
from uer import calc_map_mrr
from uer.model_builder import build_model
from uer.model_saver import save_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import BertAdam
from uer.utils.seed import set_seed
from uer.utils.tokenizer import *


class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = 6 if args.task_name == 'mlc' else 2
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.BCEWithLogitsLoss() if args.task_name == 'mlc' else nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        self.args = args
        print(f'labels num: {self.labels_num}')

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)

        logits_view = logits.view(-1, self.labels_num).float() if self.args.task_name == 'mlc' else self.softmax(
            logits.view(-1, self.labels_num))
        label_view = label.view(-1, self.labels_num).float() if self.args.task_name == 'mlc' else label.view(-1)
        loss = self.criterion(logits_view, label_view)
        return loss, logits


def re_num(s):
    s = re.sub('-?[0-9]+(\\.[0-9]+)?', '[unused1]', s)
    return s


def extract_number(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return numbers


def re_number_text(text):
    global numbers
    if isinstance(text, str):
        numbers = text.split(' ')
    elif isinstance(text, list):
        numbers = [str(t) for t in text]
    ret = ''
    for number in numbers:
        if len(number) > 4:
            number = number[:4]
        while len(number) < 4:
            number += '#'
        ret += number + ' '
    return ret


def get_multilabel_classifier_labels():
    return ["most_value", "in_decrease", "speci", "cmp", "more_less", "trend"]


def get_label_index(label):
    labels = get_multilabel_classifier_labels()
    for i in range(len(labels)):
        if label == labels[i]:
            return i
    return -1


def get_mlc_metrics(y, y_pre):
    y = np.array(y)
    y_pre = np.array(y_pre)
    assert len(y) == len(y_pre)
    assert len(y[0]) == len(y_pre[0])
    y_true_label_1_count = 0
    y_pred_label_1_count = 0
    for line in y:
        for l in line:
            if l == 1:
                y_true_label_1_count += 1
    for line in y_pre:
        for l in line:
            if l == 1:
                y_pred_label_1_count += 1
    hamming_losss = hamming_loss(y, y_pre)
    micro_f1 = f1_score(y, y_pre, average='micro')
    micro_precision = precision_score(y, y_pre, average='micro')
    micro_recall = recall_score(y, y_pre, average='micro')
    instance_f1 = f1_score(y, y_pre, average='samples')
    instance_precision = precision_score(y, y_pre, average='samples')
    instance_recall = recall_score(y, y_pre, average='samples')
    return hamming_losss, \
           micro_f1, micro_precision, micro_recall, \
           instance_f1, instance_precision, instance_recall


def add_knowledge_worker(args, params):
    p_id, path_or_sentences, columns, kg, vocab = params

    dataset = []
    if args.task_name == 'multi_choice':
        datas = json.load(open(path_or_sentences, 'r', encoding='utf-8'))
        for _, data in tqdm(enumerate(datas), total=len(datas), ncols=100):
            assert len(data) == 3
            scenario = '。'.join(data[0])[:args.scenario_max_len]
            guid = data[-1]
            for question in data[1]:
                question_text = question['question']
                choices = question['choice']
                if len(choices) != 4:
                    continue
                answer = question['answer']
                assert answer in choices
                four_labels = []
                for choice in choices:
                    label = 1 if choice == answer else 0
                    four_labels.append(label)
                    text_a = scenario + question_text
                    text_b = choice
                    text = CLS_TOKEN + text_a + SEP_TOKEN + text_b

                    text = re_num(text)
                    numbers_a = extract_number(text_a)
                    numbers_b = extract_number(text_b)

                    NUMBER_SEP_TOKEN = '[unused2]'

                    if len(numbers_a) == 0 or len(numbers_b) == 0:
                        text = text + NUMBER_SEP_TOKEN
                    else:
                        text = text + NUMBER_SEP_TOKEN + re_number_text(numbers_a) + NUMBER_SEP_TOKEN + re_number_text(
                            numbers_b)
                    text += SEP_TOKEN

                    tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True,
                                                                  max_entities=args.max_entities,
                                                                  max_length=args.seq_length, guid=[guid])
                    tokens = tokens[0]
                    pos = pos[0]
                    vm = vm[0].astype("bool")
                    token_ids = [vocab.get(t) for t in tokens]
                    mask = []
                    seg_tag = 1
                    for t in tokens:
                        if t == PAD_TOKEN:
                            mask.append(0)
                        else:
                            mask.append(seg_tag)
                        if t == SEP_TOKEN:
                            seg_tag += 1
                    dataset.append((token_ids, label, mask, pos, vm, guid, None))

    sentences_num = len(path_or_sentences)
    for line_id, line in tqdm(enumerate(path_or_sentences), total=sentences_num, desc="reading dataset", ncols=100):
        line = line.strip().split('\t')
        if True:
            if args.task_name == 'as':
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]]  # + SEP_TOKEN
                text = re_num(text)

                text_a = line[columns['text_a']]
                text_b = line[columns['text_b']]

                numbers_a = extract_number(text_a)
                numbers_b = extract_number(text_b)

                NUMBER_SEP_TOKEN = '[unused2]'

                if len(numbers_a) == 0 or len(numbers_b) == 0:
                    text = text + NUMBER_SEP_TOKEN
                else:
                    text = text + re_number_text(numbers_a) + NUMBER_SEP_TOKEN + re_number_text(numbers_b)
                text += SEP_TOKEN

                guid = line[columns["guid"]]
                template = line[columns["template"]]
                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_entities=args.max_entities,
                                                              max_length=args.seq_length, guid=[guid])
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1
                dataset.append((token_ids, label, mask, pos, vm, guid, template))
            elif args.task_name == 'mlc':
                guid = line[columns["guid"]]
                label_names = ["most_value", "in_decrease", "speci", "cmp", "more_less", "trend"]
                label = []
                for ln in label_names:
                    label.append(int(line[columns[ln.replace("_", "-")]]))

                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN
                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_entities=args.max_entities,
                                                              max_length=args.seq_length, guid=[guid])
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm, guid, None))

        # except:
        #    print("Error line: ", line)
    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="models/bert_config.json",
                        type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="word",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    # add
    parser.add_argument("--kg_rank_file", help="kg rank file",
                        default="brain/kgs/sentences_ranking_kg.rank")
    parser.add_argument("--task_name", required=True, default="as", choices=['as', 'mlc', 'multi_choice'],
                        help="task name, as or mlc")
    parser.add_argument("--labels_num", type=int, required=True, default=2, help="label num")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--mlc_test_eval_output", type=str, default='mlc_epoch_15_result')
    parser.add_argument("--no_entities", action="store_true")
    parser.add_argument("--max_entities", type=int, default=2)
    parser.add_argument("--result_output", type=str, default="output_rank.txt")
    parser.add_argument("--c3", action="store_true")
    parser.add_argument("--scenario_max_len", type=int, default=192)
    parser.add_argument("--no_label_weight", action="store_true")
    parser.add_argument("--kfold", action="store_true")
    parser.add_argument('--entities_prob', type=float, default=0.3)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--t_result_output_path', type=str)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    train_path = args.train_path.replace(os.path.basename(args.train_path), "")

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # Build classification model.
    model = BertClassifier(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            vms_batch = vms[i * batch_size: (i + 1) * batch_size]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:, :]
            vms_batch = vms[instances_num // batch_size * batch_size:]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    kg = KnowledgeGraph(args=args)

    def read_dataset(path):

        print("Loading sentences from {}".format(path))
        if args.task_name == 'multi_choice':
            params = (0, path, columns, kg, vocab)
        else:
            sentences = []
            with open(path, mode='r', encoding="utf-8") as f:
                for line_id, line in enumerate(f):
                    if line_id == 0:
                        continue
                    sentences.append(line)

            params = (0, sentences, columns, kg, vocab)

        dataset = add_knowledge_worker(args, params)

        return dataset

    def write_mlc_results(probs, guids, output_path=None):
        assert len(probs) == len(guids)
        output = open(output_path if output_path else 'mlc_test_output.txt', 'w',
                      encoding='utf-8')
        for i in range(len(probs)):
            output.write(guids[i] + '\t' + '\t'.join(list(map(lambda x: str(x), probs[i]))) + '\n')
            output.flush()
        output.close()

    def use_weight_as(probs, guids, template, weight_path=None):
        assert len(probs) == len(guids) and len(guids) == len(template)
        assert len(probs[0]) == 2
        guid_weight_map = {}
        if not weight_path:
            weight_path = args.mlc_test_eval_output
        datas = open(os.path.join(train_path, weight_path).replace('/as/', '/mlc/'), 'r',
                     encoding='utf-8').read().split(
            '\n')
        for data in datas:
            line = data.split("\t")
            guid = line[0]
            weights = list(map(lambda x: float(x), line[1:]))
            guid_weight_map[guid] = weights
        weighted_probs = []
        for ii in range(len(probs)):
            weight_index = get_label_index(template[ii])
            if weight_index == -1:
                weight = 1.0
                assert template[ii] == 'e'
            else:
                if not guids[ii] in guid_weight_map:
                    weight = 1.0
                    print('guid ' + guids[ii] + ' not found in guid weight map')
                else:
                    w = guid_weight_map[guids[ii]][weight_index]
                    # weight = math.e ** w / (math.e ** w + (1 - math.e) ** w)
                    weight = guid_weight_map[guids[ii]][weight_index] ** 0.5

            weighted_probs.append([0, probs[ii][1] * weight])
        return weighted_probs

    def calcu_map_mrr(args, is_test, probs):
        tmp_file = f'tmp/tmp_{args.learning_rate}_{args.max_entities}_{time.time()}'
        tmp_pred_output = open(tmp_file, 'w', encoding='utf-8')
        for prob in probs:
            tmp_pred_output.write(
                '\t'.join([str(prob[0]), str(prob[1])]) + '\n')
            tmp_pred_output.flush()
        tmp_pred_output.close()
        map_and_mrr_score = calc_map_mrr.cacl(tmp_file, args.test_path if is_test else args.dev_path,
                                              result_output=args.result_output, fold=args.fold)
        return map_and_mrr_score

    # Evaluation function.
    def evaluate(args, is_test, metrics='Acc', epoch=None):
        if is_test:
            dataset = read_dataset(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([example[3] for example in dataset])
        vms = [example[4] for example in dataset]
        guids = [example[5] for example in dataset]
        templates = [example[6] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()

        probs = []
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(
                batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                try:
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                except:
                    print(input_ids_batch)
                    print(input_ids_batch.size())
                    print(vms_batch)
                    print(vms_batch.size())

            logits = nn.Softmax(dim=1)(logits)
            probs += logits.tolist()

        if args.task_name == 'as':
            map_and_mrr_score = calcu_map_mrr(args, is_test, probs)
            assert len(probs[0]) == 2
            if not args.no_label_weight:
                map_and_mrr_score = calcu_map_mrr(args, is_test, use_weight_as(probs, guids, templates))
            else:
                map_and_mrr_score = calcu_map_mrr(args, is_test, probs)

            print('map: ' + str(map_and_mrr_score[0]))
            print('mrr: ' + str(map_and_mrr_score[1]))

            return {'watch_metric': 'map', 'map': map_and_mrr_score[0], 'mrr': map_and_mrr_score[1]}
        elif args.task_name == 'mlc':
            assert len(probs[0]) == 6
            predictions = []
            threshold = 1.0 / 6.0
            for prob in probs:
                pred = []
                for pp in prob:
                    pred.append(1 if pp >= threshold * 0.4 else 0)
                predictions.append(pred)
            if is_test and epoch:
                mlc_output_path = args.train_path.replace(os.path.basename(args.train_path), '')
                write_mlc_results(probs, guids,
                                  output_path=os.path.join(mlc_output_path,
                                                           f'mlc_epoch_{epoch}_result.txt'))
            else:
                write_mlc_results(probs, guids)
            hamming_loss, micro_f1, micro_precision, micro_recall, instance_f1, instance_precision, instance_recall = get_mlc_metrics(
                label_ids.tolist(), predictions)
            print(
                "hamming_loss, micro_f1, micro_precision, micro_recall, instance_f1, instance_precision, "
                "instance_recall")
            print(str((hamming_loss, micro_f1, micro_precision, micro_recall, instance_f1, instance_precision,
                       instance_recall)))
            return hamming_loss
        else:  # task_name == multi_choice
            t_results = {}
            assert len(probs[0]) == 2
            results = {}
            assert len(probs) % 4 == 0  # every question has 4 options
            correct_count = 0
            labels = label_ids.tolist()
            for i in range(0, len(probs), 4):
                if not guids[i] in t_results:
                    t_results[guids[i]] = []
                correct_label = 0
                probs_tmp = [prob[1] for prob in probs[i:i + 4]]
                if not guids[i] in results:
                    results[guids[i]] = []
                if labels[i:i + 4].index(1) == probs_tmp.index(max(probs_tmp)):
                    correct_count += 1
                    correct_label = 1
                    t_results[guids[i]].append(1)
                else:
                    t_results[guids[i]].append(0)
                results[guids[i]].append((correct_label, labels[i:i + 4].index(1), probs_tmp))
            if args.t_result_output_path:
                t_output = open(args.t_result_output_path, 'w', encoding='utf-8')
                t_output.write(json.dumps(t_results))
                t_output.flush()
                t_output.close()
            acc = 4 * correct_count / len(probs)
            return {'watch_metric': 'acc', 'acc': acc}

    if args.do_eval and not args.do_train:
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        set_seed(args.seed)
        result = evaluate(args, True)
        print(str(result))
        return

    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path)
    print("Shuffling dataset")
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    print("Trans data to tensor.")
    print("input_ids")
    input_ids = torch.LongTensor([example[0] for example in trainset])
    print("label_ids")
    label_ids = torch.LongTensor([example[1] for example in trainset])
    print("mask_ids")
    mask_ids = torch.LongTensor([example[2] for example in trainset])
    print("pos_ids")
    pos_ids = torch.LongTensor([example[3] for example in trainset])
    print("vms")
    vms = [example[4] for example in trainset]

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    result = 0.0
    best_result = 0.0

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(
                batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
            model.zero_grad()

            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))
                sys.stdout.flush()
                total_loss = 0.
            loss.backward()
            optimizer.step()

        # save_model(model, args.output_model_path)

        print("Start evaluation on fold dev dataset.")
        result_dev = evaluate(args, False)
        print("Start evaluation on fold test dataset.")
        result_test = evaluate(args, True, epoch=epoch)
        print(f'epoch {epoch}, dev: {str(result_dev)}, test: {str(result_test)}\n\n')
        result = result_dev
        # save_model(model, args.output_model_path)
        if args.task_name == 'mlc':
            continue
        if result[result['watch_metric']] > best_result:
            best_result = result[result['watch_metric']]
            save_model(model, args.output_model_path)
        if args.c3:
            save_model(model, args.output_model_path)

    # Evaluation phase.
    print("Final evaluation on the test dataset.")
    set_seed(args.seed)
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))
    result = evaluate(args, True)
    print(f'results: {str(result)}')


if __name__ == "__main__":
    main()
