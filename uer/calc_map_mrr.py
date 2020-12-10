import json


def ranking(outputs, all_t_count_map=None):
    predictions = [[o[1]] for o in outputs['prob']]
    targets = outputs['target']
    assert len(predictions) == len(targets)
    t_count = 0
    for i, pred in enumerate(predictions):
        if targets[i] == 1:
            t_count += 1
        predictions[i].append(targets[i])
    predictions.sort(key=lambda x: x[0], reverse=True)
    map_sum = 0.0
    mrr_sum = -1.0
    count = 1
    for i, pred in enumerate(predictions):
        if pred[1] == 1:
            map_sum += float(count) / (i + 1)
            count += 1
            if mrr_sum == -1:
                mrr_sum = 1.0 / (i + 1)

    if all_t_count_map:
        t_count = all_t_count_map[outputs['guid']]

    if t_count == 0:
        map_ = 0.0
        mrr = 0.0
    else:
        map_ = map_sum / t_count
        mrr = mrr_sum
    return {
        'map_': map_,
        'mrr_': mrr,
    }


def get_all_t_count(test_file):
    datas = _read_tsv(test_file)
    t_count_map = {}
    for data in datas:
        guid = data[3]
        label = data[2]
        if not guid in t_count_map:
            t_count_map[guid] = 0
        if label == '1':
            t_count_map[guid] = t_count_map[guid] + 1
    return t_count_map


def _read_tsv(input_file, _format=str, is_result=False):
    f = open(input_file)
    lines = f.read().split('\n')
    ret = []
    for line in lines:
        if 'text_a' in line and 'text_b' in line:
            continue
        if line.strip() == '':
            continue
        ll = line.split('\t')
        if is_result:
            ll = ll[:2]
        ret.append(list(map(lambda x: _format(x), ll)))
    return ret


def write_line(lines, output):
    output.write(lines[0][0].split('\t')[3] + '\n' + '-' * 20 + '\n')
    for line in lines:
        line_split = line[0].split('\t')
        new_line = '\t'.join([line_split[2], line_split[1], line[0]])
        output.write(new_line + '\t' + str(line[1]) + '\n')
        # output.write(line_split[1] + '\n')
        output.flush()
    output.write('\n\n')


def cacl(result_file=None, test_file=None, use_all_t_count=False, result_output=None, fold=None):
    results = _read_tsv(result_file, float, is_result=True)
    tests = _read_tsv(test_file)
    if len(results) != len(tests):
        print('result file: ' + result_file)
        print('test file: ' + test_file)
        print('results len: ' + str(len(results)))
        print('test len: ' + str(len(tests)))
    assert len(results) == len(tests)
    guid = tests[0][3]
    prob = []
    target = []
    line = []
    t_count = 0
    map_sum = 0.0
    mrr_sum = 0.0
    t_result = f't{fold}_no_entities.json'
    tt_result = {}
    if use_all_t_count:
        all_t_count_map = get_all_t_count(test_file)
    else:
        all_t_count_map = None
    if result_output:
        output_rank = open(result_output, 'w', encoding='utf-8')
    for i, result in enumerate(results):
        if tests[i][3] == guid:
            prob.append(result)
            target.append(int(tests[i][2]))
            line.append(('\t'.join(tests[i]), result[1]))
        else:
            outputs = {}
            outputs['guid'] = tests[i - 1][3]
            outputs['prob'] = prob
            outputs['target'] = target
            line.sort(key=lambda x: x[1], reverse=True)
            if result_output:
                write_line(line, output_rank)
            t_count += 1
            mm = ranking(outputs, all_t_count_map=all_t_count_map)

            map_sum += mm['map_']
            mrr_sum += mm['mrr_']
            tt_result[guid] = (mm['map_'], mm['mrr_'])
            guid = tests[i][3]
            prob = [result]
            target = [int(tests[i][2])]
            line = [('\t'.join(tests[i]), result[1])]

    line.sort(key=lambda x: x[1], reverse=True)
    if result_output:
        write_line(line, output_rank)
    t_count += 1
    mm = ranking(outputs, all_t_count_map=all_t_count_map)
    map_sum += mm['map_']
    mrr_sum += mm['mrr_']
    tt_result[guid] = (mm['map_'], mm['mrr_'])
    output = open(t_result, 'w')
    output.write(json.dumps(tt_result))
    output.flush()
    output.close()

    if result_output:
        output_rank.close()
    return map_sum / t_count, mrr_sum / t_count
