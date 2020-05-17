import json
from tqdm import tqdm
from sys import exit
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
    path = DATA_DIR + "mt_arggen_20/test.jsonl.filtered"
    dataset = {"op": [], "passages": [], "inners": [], "speaker": [], "passage_kp": [], "id": [], "targetId": []}

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
        cur_inner_speaker = list()
        for utter in cur_obj['inners']:
            cur_inner_set.append([utter['body']])
            cur_inner_speaker.append(utter['isOP'])
        dataset["inners"].append(cur_inner_set)
        dataset["speaker"].append(cur_inner_speaker)
        
    logging.info("Arggen test data loaded. %d samples in total." % (len(dataset["id"])))
    return dataset


def _load_wikigen_test_data(demo=False):
    """
    Load test data for wikipedia paragraph generation task.
    """
    path = DATA_DIR + "wikigen/test.jsonl"
    dataset = {"title": [], "style": [], "ph_bank": []}

    logging.info("Loading test data for wikigen...")
    raw_lns = open(path).readlines()
    if demo:
        raw_lns = raw_lns[:100]

    for ln in tqdm(raw_lns):
        cur_obj = json.loads(ln)
        dataset["title"].append(cur_obj["title"])
        dataset["ph_bank"].append(cur_obj["ph_bank"])
        dataset["style"].append(1)

        dataset["title"].append(cur_obj["title"])
        dataset["ph_bank"].append(cur_obj["ph_bank"])
        dataset["style"].append(0)
    return dataset


def _load_absgen_test_data(demo=False):
    """
    Load test data for abstract generation task.
    """
    path = DATA_DIR + "absgen/test.jsonl"
    dataset = {"title": [], "ph_bank": []}
    logging.info("Loading test data for absgen...")
    raw_lns = open(path).readlines()
    if demo:
        raw_lns = raw_lns[:100]

    for ln in tqdm(raw_lns):
        cur_obj = json.loads(ln)
        dataset["title"].append(cur_obj["title"])
        dataset["ph_bank"].append(cur_obj["ph_bank"])

    return dataset


def _load_absgen_train_data(demo=False):
    """
    Load abstract generation training and dev data.
    Args:
        demo: bool. If set to True only load 100 samples.
    Returns:
        dataset:
    """
    dataset = {set_type: {"tgt": [], "title": [], "kp_sel": [], "ph_bank": []} \
               for set_type in ['train', 'dev']}

    for set_type in dataset:
        path = DATA_DIR + "absgen/%s.jsonl" % set_type

        for ln in open(path):
            cur_obj = json.loads(ln)

            dataset[set_type]["title"].append(cur_obj["title"])
            dataset[set_type]["ph_bank"].append(cur_obj["ph_bank"])
            dataset[set_type]["kp_sel"].append(cur_obj["ph_sel"])
            dataset[set_type]["tgt"].append(cur_obj["abstract_words"])

            if demo and len(dataset[set_type]["title"]) == 100:
                break
    print("Abstract data loaded. train/dev=%d/%d" % (len(dataset["train"]["title"]), len(dataset["dev"]["title"])))
    return dataset


def _load_wiki_train_data(demo=False):
    """
    Load Wikipedia generation training and dev data.
    Args:
        demo: bool. If set to True only load 100 samples.
    Returns:
        dataset:
    """
    dataset = {set_type: {"tgt": [], "title": [], "kp_sel": [], "style": [], "ph_bank": []} \
               for set_type in ["train", "dev"]}

    for set_type in dataset:
        if demo:
            set_type = "train"
        path = DATA_DIR + "wikigen/%s.jsonl" % set_type

        for ln in open(path):
            cur_obj = json.loads(ln)
            """ 
            "title": a list of string for the article title
                e.g. ["septic", "tank"]
            "sents": a list of sentences, each sentence is a list of words
                e.g. "sents": [["a", "septic", "tank", "is", "an", "underground", "chamber", "made", ...],
                               ["settling", "and", "anaerobic", "processes", "reduce", "solids", "and", ...],
                               ["septic", "tank", "systems", "are", "a", "type", "of", "simple",...]]
            "ph_sel": a list of phrases for each sentence, where each phrase is a list of words
                e.g. "ph_sel": [[["flows", "for", "basic", "treatment"], ["domestic", "wastewater"], ...],
                                [["moderate"], ["reduce", "solides"], ["anaerobic", "processes"], ...]]
            """
            dataset[set_type]["title"].append(cur_obj["title"])
            dataset[set_type]["tgt"].append(cur_obj["normal_sents"])
            dataset[set_type]["kp_sel"].append(cur_obj["normal_ph_sel"])
            dataset[set_type]["style"].append(1)
            dataset[set_type]["ph_bank"].append(cur_obj["ph_bank"])

            dataset[set_type]["title"].append(cur_obj["title"])
            dataset[set_type]["tgt"].append(cur_obj["simple_sents"])
            dataset[set_type]["kp_sel"].append(cur_obj["simple_ph_sel"])
            dataset[set_type]["style"].append(0)
            dataset[set_type]["ph_bank"].append(cur_obj["ph_bank"])

            if demo and len(dataset[set_type]["title"]) >= 100:
                break
    print("Wikipedia data loaded, train/dev=%d/%d" % (len(dataset["train"]["title"]), len(dataset["dev"]["title"])))
    return dataset


def _load_arggen_train_data(demo=False):
    """
    Load training and validation data. Data format is detailed below:
    `op` (list):  tokenized OP
    `target_counterarg` (list): a list of sentences in root reply (target argument)
    `target_retrieved_passages` (list): a list of retrieved passages, which contains sentences and keyphrases
    """
    dataset = dict()
    dataset["train"] = {"src": {"op": [], "passages": [], "speaker": [], "passage_kp": [], "inners": []},
                        "tgt": [],
                        "id": []}

    dataset["dev"] = {"src": {"op": [], "passages": [], "speaker": [], "passage_kp": [], "inners": []},
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
            raw_lns = open(DATA_DIR + "mt_arggen_20/%s.jsonl.op" % set_type).readlines()

        for ln in tqdm(raw_lns):
            cur_obj = json.loads(ln)
            ln_cnt += 1

            dataset[set_type]["src"]["op"].append(cur_obj["op"])
            dataset[set_type]["id"].append(cur_obj["id"])
            dataset[set_type]["tgt"].append(cur_obj["target_counterarg"])

            cur_passage_set = list()
            cur_passage_kp_set = list()
            for psg in cur_obj["target_retrieved_passages"]:
                # cur_passage_set.append(psg["sentences"])
                cur_passage_set.append([])
                cur_passage_kp_set.append(psg["keyphrases"])
            dataset[set_type]["src"]["passages"].append(cur_passage_set)
            dataset[set_type]["src"]["passage_kp"].append(cur_passage_kp_set)
  
            cur_inner_set = list()
            cur_inner_speaker = list()
            for utter in cur_obj['inners']:
                cur_inner_set.append([utter['body']])
                cur_inner_speaker.append(utter['isOP'])
            dataset[set_type]["src"]["inners"].append(cur_inner_set)
            dataset[set_type]["src"]["speaker"].append(cur_inner_speaker)

            if demo and ln_cnt >= 100:
                break

        logging.info("%s data loaded, %d samples in total" % (set_type, ln_cnt))

    return dataset