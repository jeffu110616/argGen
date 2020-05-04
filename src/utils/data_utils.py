import json
from tqdm import tqdm
import logging
from utils.misc_utils import DATA_DIR

def load_test_data(task="arggen", demo=False, opt=None):
    if task == "arggen":
        return _load_arggen_test_data(demo=demo, opt=opt)
    elif task == "wikigen":
        return _load_wikigen_test_data(demo=demo)
    else:
        return _load_absgen_test_data(demo=demo)

def load_train_data(demo=False, task=None):
    assert task is not None, "Task has to be specified!"
    if task == "wikigen":
        return _load_wiki_train_data(demo=demo)
    elif task == "arggen":
        return _load_arggen_train_data(demo=demo)
    elif task == "absgen":
        return _load_absgen_train_data(demo=demo)
    else:
        raise ValueError("Specified task {} does not exist!".format(task))


def _load_arggen_test_data(demo=False, opt=None):
    """
    Load test data for argument generation task.
    """
    path = DATA_DIR + "mt_arggen_20/test.jsonl"
    dataset = {"op": [], "passages": [], "inners": [], "passage_kp": [], "id": [], "targetId": []}

    logging.info("Loading test data for arggen...")
    raw_lns = open(path).readlines()
    if demo:
        raw_lns = raw_lns[:100]

    if opt.infer_fold != -1 and opt.infer_fold_selected != -1:
        assert(opt.infer_fold_selected > 0 and opt.infer_fold_selected <= opt.infer_fold)
        sectionLength = (len(raw_lns) // opt.infer_fold)+1
        startIdx = sectionLength * (opt.infer_fold_selected - 1)
        endIdx = sectionLength * (opt.infer_fold_selected)
        endIdx = endIdx if endIdx <= len(raw_lns) else len(raw_lns)
        raw_lns = raw_lns[startIdx:endIdx]
        print("Total partition: {}, startIdx: {}, endIdx: {}, {} examples in total.".format(opt.infer_fold, startIdx, endIdx, len(raw_lns)))

    for ln in tqdm(raw_lns):
        cur_obj = json.loads(ln)
        dataset["op"].append(cur_obj["op"])
        dataset["id"].append(cur_obj["id"])
        dataset["targetId"].append(cur_obj["targetId"])
        cur_passage_sent_lst = []
        cur_passage_kp_set = set()

        for psg in cur_obj["op_retrieved_passages"]:
            cur_passage_sent_lst.append(psg["sentences"])
            for kp in psg["keyphrases"]:
                cur_passage_kp_set.add(kp)


        dataset["passages"].append(cur_passage_sent_lst)
        dataset["passage_kp"].append(cur_passage_kp_set)

        cur_inner_set = list()
        for utter in cur_obj['inners']:
            cur_inner_set.append(utter)
        dataset["inners"].append(cur_inner_set)

    logging.info("Arggen test data loaded. %d samples in total." % (len(dataset["id"])))
    return dataset

def _load_arggen_train_data(demo=False):
    """
    Load training and validation data. Data format is detailed below:
    `op` (list):  tokenized OP
    `target_counterarg` (list): a list of sentences in root reply (target argument)
    `target_retrieved_passages` (list): a list of retrieved passages, which contains sentences and keyphrases
    """
    dataset = dict()
    dataset["train"] = {"src": {"op": [], "passages": [], "passage_kp": [], "inners": []},
                        "tgt": [],
                        "id": []}

    dataset["dev"] = {"src": {"op": [], "passages": [], "passage_kp": [], "inners": []},
                      "tgt": [],
                      "id": []}

    for set_type in ["train", "dev"]:
        ln_cnt = 0
        logging.info("loading %s data..." % set_type)

        if demo:
            # raw_lns = open(DATA_DIR + "arggen/train.jsonl").readlines()
            raw_lns = open(DATA_DIR + "mt_arggen_20/train.jsonl").readlines()
            raw_lns = raw_lns[:10]
        else:
            # raw_lns = open(DATA_DIR + "arggen/%s.jsonl" % set_type).readlines()
            raw_lns = open(DATA_DIR + "mt_arggen_20/%s.jsonl" % set_type).readlines()

        for ln in tqdm(raw_lns):
            cur_obj = json.loads(ln)
            ln_cnt += 1

            dataset[set_type]["src"]["op"].append(cur_obj["op"])
            dataset[set_type]["id"].append(cur_obj["id"])
            dataset[set_type]["tgt"].append(cur_obj["target_counterarg"])

            cur_passage_set = list()
            cur_passage_kp_set = list()
            for psg in cur_obj["target_retrieved_passages"]:
                cur_passage_set.append(psg["sentences"])
                cur_passage_kp_set.append(psg["keyphrases"])
            dataset[set_type]["src"]["passages"].append(cur_passage_set)
            dataset[set_type]["src"]["passage_kp"].append(cur_passage_kp_set)
  
            cur_inner_set = list()
            for utter in cur_obj['inners']:
                cur_inner_set.append(utter)
            dataset[set_type]["src"]["inners"].append(cur_inner_set)

            if demo and ln_cnt >= 100:
                break

        logging.info("%s data loaded, %d samples in total" % (set_type, ln_cnt))

    return dataset