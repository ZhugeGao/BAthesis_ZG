import math
import statistics
import random

import pandas
import torch
import tensorflow as tf
from transformers import BertTokenizer, BertForMaskedLM
import csv
import time
import numpy as np
from transformers import *
import torch.nn.functional as F
from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm
import copy
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, SGDClassifier
from sklearn.svm import LinearSVC, SVR
import pandas as pd
from collections import Counter
from tqdm import tqdm
import pickle
import jsonlines
from statistics import mean
import argparse
import collections
import spacy
import time
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 115
plt.rcParams['figure.figsize'] = [8.0, 6.0]
import os
from collections import OrderedDict


def sort_vocab(embed, output_embeddings):
    best = (np.argsort(np.dot(output_embeddings, embed)))[::-1]
    return list(best)


def sort_vocab_batch(embed, output_embeddings):
    best = (np.argsort(np.dot(output_embeddings, embed.T),
                       axis=0))  # notes: dot product of all the embeddings and one embedding. rank it in ascending order. The dot product of two normalized (unit) vectors will be a scalar value between -1 and 1.
    return list(best.T)


def sort_vocab_with_log_freq(embed, output_embeddings, tokenizer_mlm, word_log_freq_dict, binary_freq=False,
                             norm_vec=True, square=True, norm_cos_sim_list=False, mean=None):
    assert mean in [None, 'geometric', 'harmonic'], 'mean could only be None, geometric or harmonic'

    # start_time = time.time()
    # print("sorting vocab")
    # print(output_embeddings[0])
    if norm_vec:
        cos_sim_list = np.empty(len(output_embeddings))
        for i, emb in enumerate(output_embeddings):
            cos_sim_list[i] = np.dot(emb, embed.T) / (norm(emb) * norm(embed))
            if square:
                cos_sim_list[i] = np.sign(cos_sim_list[i]) * np.abs(cos_sim_list[i]) * np.abs(cos_sim_list[i])

    else:
        cos_sim_list = np.dot(output_embeddings, embed.T)  # transform to 0-1. normalize each pair

    # print("finish calculating cos similarity list", time.time()-start_time)
    print("min_cos_sim: ", np.min(cos_sim_list))
    print("max_cos_sim: ", np.max(cos_sim_list))
    if norm_cos_sim_list:
        min_val = np.min(cos_sim_list)
        max_val = np.max(cos_sim_list)
        for i, cos_sim in enumerate(cos_sim_list):
            cos_sim_list[i] = (cos_sim - min_val) / (max_val - min_val)

        print("min_cos_sim: (normalizing check)", np.min(cos_sim_list))
        print("max_cos_sim: (normalizing check)", np.max(cos_sim_list))

    # print(cos_sim_list[0])
    # print(cos_sim_list[0].shape)
    # cnt = 0

    for i, cos_sim in enumerate(cos_sim_list):
        word_token = tokenizer_mlm.convert_ids_to_tokens(i)
        if word_token in word_log_freq_dict:
            rel_log_freq = word_log_freq_dict[word_token]
            if binary_freq:
                rel_log_freq = 1.0
            # print("")
            # print(cos_sim_list[i])

            # geometric mean
            # print(cos_sim * rel_log_freq)
            if mean == "geometric":
                cos_sim_list[i] = math.sqrt(cos_sim * rel_log_freq)  # store the inverse log freq and use multiplication
            # harmonic mean
            if mean == "harmonic":
                cos_sim_list[i] = (cos_sim * rel_log_freq * 2.0) / (cos_sim + rel_log_freq)  # change into np.array
            if mean is None:
                cos_sim_list[i] = (cos_sim + rel_log_freq) / 2
            # print(cos_sim_list[i])

        else:
            # cos_sim_list[i] = np.divide(cos_sim, np.inf)
            cos_sim_list[i] = 0.0  # the value...?
            # cos_sim_list[i] = np.divide(np.multiply(2, np.multiply(cos_sim_list[i], np.inf)),
            #                             np.sum(cos_sim_list[i], np.inf))
    # print("get best ranking", time.time()-start_time)
    best = list(np.argsort(cos_sim_list))[::-1]  # , axis=0
    # plt.plot(best)
    # plt.show()
    # y_max = np.max(cos_sim_list)
    # x_max = np.argmax(cos_sim_list)

    return best, cos_sim_list  # index .T


def average(lst):
    return sum(lst) / len(lst)


def compute_log_frequency_for_lang(lang_word_list_path, base=10, sqrt=False, normalize=False):
    # print("computing log frequency")
    # start_time = time.time()
    # read word list and count them
    word_list = []
    # total = 0
    word_rel_log_freq = OrderedDict()
    # word_log_freq = {}
    with open(lang_word_list_path) as words:
        reader = csv.reader(words, delimiter=" ")
        for line in reader:
            word_list.append((line[0], int(line[1])))
            # total += int(line[1])
    cnt = Counter(dict(word_list))
    total = sum(cnt.values())
    for word, count in cnt.most_common():
        # var names
        rel_freq = count / total

        log_freq = -math.log(rel_freq, base)
        inv_log_freq = 1 / log_freq
        if sqrt:
            inv_log_freq = math.sqrt(inv_log_freq)
        word_rel_log_freq[word] = inv_log_freq
        # print(w[0])
    # print(total)
    # if sqrt and normalize:
    #     min_val = min(word_rel_log_freq.values())
    #     max_val = max(word_rel_log_freq.values())
    #     for key, value in word_rel_log_freq.items():
    #         word_rel_log_freq[key] = (value - min_val) / (max_val - min_val)
    # print(word_rel_log_freq)
    # print("finished", time.time()-start_time)
    return word_rel_log_freq


def get_average_vec_for_category(source_word, word_pair_list, tokenizer_mlm, output_embeddings):
    # sub avg all the word pair with identical first word, for other word_pair without sea
    # leave out the word pairs with identical first word, when first word is sea
    new_emb_list = []
    new_emb_dict = {}
    for w1, w2 in word_pair_list:
        if w1 == source_word:
            continue

        w1_id = tokenizer_mlm.encode(w1, add_special_tokens=False)
        w1_emb = output_embeddings[w1_id]
        if (len(w1_id) != 1):
            # print("w1" + w1)
            continue
        w2_id = tokenizer_mlm.encode(w2, add_special_tokens=False)
        w2_emb = output_embeddings[w2_id]
        if (len(w2_id) != 1):
            # print("w2" + w2)
            continue
        new_emb = w2_emb - w1_emb
        if w1 not in new_emb_dict:
            new_emb_dict[w1] = []
        new_emb_dict[w1].append(new_emb)
        # new_emb_list.append(new_emb)
    for k, v in new_emb_dict.items():
        new_emb_list.extend(np.average(v, axis=0))
    return np.average(new_emb_list, axis=0)


def print_top_n_result(rank, tokenizer_mlm, n=20):
    output = ""
    for result_id in list(rank)[:n - 1]:
        result_id = result_id.item()
        # print(type(result_id))
        output += tokenizer_mlm.convert_ids_to_tokens(result_id) + ", "
        # print(len(tokenizer_mlm.decode(result_id)))
    print(output)
    print("")


def get_index_for_word_list(word_list, tokenizer):
    freq_indexes = []
    for w in word_list:
        id = tokenizer.encode(w, add_special_tokens=False)
        if len(id) != 1: continue
        id = id[0]
        freq_indexes.append(id)
    return freq_indexes


def plot_annotate_top_n(source_word, target_word, best, cos_sim_list, word_list, marker_size=1, annotate_top_n=10,
                        show_top_n=10, sort_cos_sim=False, tokenizer=None, show_cos_annotation=False,
                        parameters=[]):  # title, words
    assert show_top_n in [None, 10, 50, 100, 1000, 10000], 'please input an int or None'
    if show_top_n is None:
        show_top_n = len(best) + 1
    cos_sim_dict = get_top_n_cos_sim(best, cos_sim_list, n=show_top_n)
    best = best[:show_top_n - 1]
    ordered_cos_sim = [cos_sim_dict[i] for i in best]  # sorted list of cos sim values
    s = [marker_size] * len(cos_sim_dict)

    freq_indexes = get_index_for_word_list(word_list, tokenizer=tokenizer)
    # highlights_dict = {k:v for k,v in freq_dict.items() if k in cos_sim_dict}
    if sort_cos_sim:
        # sort by cos sim
        highlights_x = [i for i, original_index in enumerate(best) if original_index in freq_indexes]
        print("number of " + str(len(word_list)) + " most frequent words in " + str(show_top_n) + ": " + str(
            len(highlights_x)))
        if len(highlights_x) > 0:
            print("mean: " + str(statistics.mean(highlights_x)))
            print("median: " + str(statistics.median(highlights_x)))
            print("frequent word ranks: " + str(highlights_x))
        highlights_y = [cos_sim_list[best[i]] for i in highlights_x]
        plt.scatter(list(range(len(best))), ordered_cos_sim, s=s)
        plt.xlabel("sorted by cos sim")
    else:
        # original index order
        highlights_x = [i for i in freq_indexes if i in cos_sim_dict]
        highlights_y = [cos_sim_list[i] for i in highlights_x]
        plt.scatter(cos_sim_dict.keys(), cos_sim_dict.values(), s=s)
        plt.xlabel("original Bert index")

    highlights_s = [marker_size * 2] * len(highlights_x)

    # sort by cos sim

    plt.scatter(highlights_x, highlights_y, s=highlights_s, color='red')
    plt.ylabel("cos sim")
    title = source_word + ">" + target_word + ",top_" + str(show_top_n) + "," + parameters[-1]
    # for arg in parameters:
    #     title += " " + arg + " "
    dir = ""
    for p in parameters[:-1]:
        dir += p + ","
    plt.title(title + ',' + dir)

    for i, index in enumerate(best[:annotate_top_n - 1]):
        # change annotate to also work for sorted version
        text = index
        x = index
        if tokenizer is not None:
            if sort_cos_sim:
                # sort cos sim
                text = tokenizer.convert_ids_to_tokens(index.item())
                x = i
            else:
                # original
                text = tokenizer.convert_ids_to_tokens(index.item())
        if show_cos_annotation:
            text = "(" + text + ", " + str(np.around(cos_sim_dict[index], decimals=5)) + ")"
        plt.annotate(text, xy=(x, cos_sim_dict[index]), fontsize=2)
    img_path = "../data/" + dir + '/' + title + '.svg'  # '.png'
    fig = plt.gcf()
    fig.savefig(img_path)
    plt.show()


def get_top_n_cos_sim(best, cos_sim_list, n=1000):
    new_cos_dict = {}
    for i in best[:n - 1]:
        new_cos_dict[i] = cos_sim_list[i]
    return new_cos_dict


# def printLog(*args, **kwargs):
#     print(*args, **kwargs)
#     with open(output,'a') as file:
#         print(*args, **kwargs, file=file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_data", action="store_true",
                        help="extract data from scratch (otherwise, load data from saved file)")
    parser.add_argument("--all_langs", action="store_true", help="Translate from all langs to all langs")
    parser.add_argument("--eval_pos", help="evaluate a specific pos tag: NOUN, VERB or ADJ")
    args = parser.parse_args()

    run_word_pair = True
    run_category = True

    show = [50, 100, 1000, 10000, None]  # 10,
    sort_cos_sim = True
    # sort_cos_sim = False

    mean = None
    # mean = 'geometric'  # 'geometric' 'harmonic' None
    # mean = 'harmonic' # not good, try with square.
    # norm_cos=False
    norm_cos = True
    # binary=False
    binary = True
    # square=True
    square = False

    annotate = 50
    highlight = 200
    parameters = ["mean=" + str(mean), "norm_cos=" + str(norm_cos), "binary=" + str(binary), "square=" + str(square)]

    dir = "../data/"
    for p in parameters:
        dir += p + ","
    if not os.path.exists(dir):
        os.mkdir(dir)
    # load mBERT
    # make an argument to switch between bert and multilingual bert
    # print("loading bert")
    # start_time = time.time()
    pretrained_weights = 'bert-base-uncased'  # -multilingual
    tokenizer_mlm = BertTokenizer.from_pretrained(pretrained_weights)
    model_mlm = BertForMaskedLM.from_pretrained(pretrained_weights, output_hidden_states=True)
    output_embeddings = model_mlm.cls.predictions.decoder.weight.detach().cpu().numpy()
    # print("bert is loaded", time.time()-start_time)
    # normalize embeddings
    # sum_of_rows = output_embeddings.sum(axis=1)
    # output_embeddings = output_embeddings / sum_of_rows[:, np.newaxis]

    # a1, a2, b1, b2 = ["man", "king", "woman", "queen"]
    # a1, a2, b1, b2 = ["hand", "glove", "shoe", "feet"]
    # a1, a2, b1, b2 = ["hand", "glove", "feet", "shoe"]
    # a1, a2, b1, b2 = ["hand", "glove", "foot", "shoe"]
    # a1, a2, b1, b2 = ["hands", "glove", "feet", "shoe"]
    # a1, a2, b1, b2 = ["hands", "gloves", "feet", "shoes"]
    b1_word_ranks = {}
    b2_word_ranks = {}
    category_word_ranks = {}

    word_log_freq_dict = compute_log_frequency_for_lang(
        "/Users/zhugegao/PycharmProjects/BAthesis_ZG/data/en/en_50k.txt", sqrt=False)
    word_list = list(word_log_freq_dict.keys())
    word_list = word_list[:highlight]

    if run_word_pair:
        with open("../data/word_lists/analogy_word_pairs.csv") as words:
            reader = csv.reader(words)
            baseline_b1_list = []
            baseline_b2_list = []
            rank_b1_list = []
            rank_b2_list = []
            b1_list = []
            b2_list = []
            for row in reader:
                a1, a2, b1, b2 = row
                a1_id = tokenizer_mlm.encode(a1, add_special_tokens=False)
                if len(a1_id) != 1: continue
                a1_emb = output_embeddings[a1_id]

                a2_id = tokenizer_mlm.encode(a2, add_special_tokens=False)
                if len(a2_id) != 1: continue
                a2_emb = output_embeddings[a2_id]

                b1_id = tokenizer_mlm.encode(b1, add_special_tokens=False)
                if len(b1_id) != 1: continue
                b1_emb = output_embeddings[b1_id]

                b2_id = tokenizer_mlm.encode(b2, add_special_tokens=False)
                if len(b2_id) != 1: continue
                b2_emb = output_embeddings[b2_id]

                new_b1_emb = a1_emb - a2_emb + b2_emb

                print("a1, a2, b1, b2 =[", a1, a2, b1, b2, "]")
                # b1_id_ranks = sort_vocab_batch(new_b1_emb, output_embeddings)

                b1_baseline_rank, _ = sort_vocab_with_log_freq(b2_emb, output_embeddings, tokenizer_mlm,
                                                               word_log_freq_dict,
                                                               binary_freq=binary, norm_vec=True, square=square,
                                                               norm_cos_sim_list=norm_cos,
                                                               mean=mean)

                baseline_output = ""
                baseline_b1_list.append(list(b1_baseline_rank).index(b1_id) + 1)
                # b1_word_ranks[b1] = rank_b1_list[-1]
                print(b1 + " baseline rank: " + str(baseline_b1_list[-1]))  # [0]
                print_top_n_result(b1_baseline_rank, tokenizer_mlm)

                b1_id_ranks, b1_cos_list = sort_vocab_with_log_freq(new_b1_emb, output_embeddings, tokenizer_mlm,
                                                                    word_log_freq_dict,
                                                                    binary_freq=binary, norm_vec=True, square=square,
                                                                    norm_cos_sim_list=norm_cos,
                                                                    mean=mean)
                parameters_b1 = parameters.copy()
                parameters_b1.append("(" + a2 + " > " + a1 + ")")

                for s in show:
                    plot_annotate_top_n(b2, b1, b1_id_ranks, b1_cos_list, word_list=word_list, annotate_top_n=annotate,
                                        show_top_n=s, sort_cos_sim=sort_cos_sim,
                                        tokenizer=tokenizer_mlm, parameters=parameters_b1)

                rank_b1_list.append(list(b1_id_ranks).index(b1_id) + 1)
                b1_list.append(b1)
                b1_word_ranks[b1] = rank_b1_list[-1]
                print(b1 + " rank: " + str(rank_b1_list[-1]))  # [0]
                print_top_n_result(b1_id_ranks, tokenizer_mlm)

                print("")

                new_b2_emb = a2_emb - a1_emb + b1_emb
                # b2_id_ranks = sort_vocab_batch(new_b2_emb, output_embeddings)
                b2_baseline_rank, _ = sort_vocab_with_log_freq(b1_emb, output_embeddings, tokenizer_mlm,
                                                               word_log_freq_dict,
                                                               binary_freq=binary, norm_vec=True, square=square,
                                                               norm_cos_sim_list=norm_cos,
                                                               mean=mean)

                baseline_b2_list.append(list(b2_baseline_rank).index(b2_id) + 1)
                # b1_word_ranks[b1] = rank_b1_list[-1]
                print(b2 + " baseline rank: " + str(baseline_b2_list[-1]))  # [0]
                print_top_n_result(b2_baseline_rank, tokenizer_mlm)

                parameters_b2 = parameters.copy()
                parameters_b2.append("(" + a1 + " > " + a2 + ")")
                b2_id_ranks, b2_cos_list = sort_vocab_with_log_freq(new_b2_emb, output_embeddings, tokenizer_mlm,
                                                                    word_log_freq_dict,
                                                                    binary_freq=binary, norm_vec=True, square=square,
                                                                    norm_cos_sim_list=norm_cos,
                                                                    mean=mean)
                for s in show:
                    plot_annotate_top_n(b1, b2, b2_id_ranks, b2_cos_list, word_list=word_list, annotate_top_n=annotate,
                                        show_top_n=s, sort_cos_sim=sort_cos_sim,
                                        tokenizer=tokenizer_mlm, parameters=parameters_b2)
                rank_b2_list.append(list(b2_id_ranks).index(b2_id) + 1)
                b2_list.append(b2)
                b2_word_ranks[b2] = rank_b2_list[-1]
                print(b2 + " rank: " + str(list(b2_id_ranks).index(b2_id) + 1))
                print_top_n_result(b2_id_ranks, tokenizer_mlm)

                # print("avg b2: " + str(average(b2_id_ranks)))`

        if run_category:
            categories_word_list = ["instrument", "instrument_inv", "crime", "crime_inv", "growth", "growth_inv",
                                    "profession", "profession_inv"]
            category_output = ""
            for category in categories_word_list:
                t_win = 0
                word_ranks = {}
                word_pair_list = []
                baseline_list = []
                with open("../data/word_lists/{}.csv".format(category)) as words:
                    print(category)
                    reader = csv.reader(words)
                    for row in reader:
                        w1, w2 = row
                        word_pair_list.append([w1, w2])

                # leave one out: using a for loop
                leave_one_out_list = [[e for e in word_pair_list if e != word_pair_list[i]] for i in
                                      range(len(word_pair_list))]

                for i, l in enumerate(leave_one_out_list):
                    w = word_pair_list[i][0]
                    w_id = tokenizer_mlm.encode(w, add_special_tokens=False)
                    if len(w_id) != 1: continue
                    w_emb = output_embeddings[w_id]

                    t = word_pair_list[i][1]
                    t_id = tokenizer_mlm.encode(t, add_special_tokens=False)
                    if len(t_id) != 1: continue

                    avg_vec = get_average_vec_for_category(w, l, tokenizer_mlm, output_embeddings)

                    t_emb_new = w_emb + avg_vec
                    print(w, t)
                    baseline_ranks, _ = sort_vocab_with_log_freq(w_emb, output_embeddings, tokenizer_mlm,
                                                                 word_log_freq_dict, binary_freq=binary, norm_vec=True,
                                                                 square=square, norm_cos_sim_list=norm_cos, mean=mean)
                    baseline_list.append(list(baseline_ranks).index(t_id) + 1)
                    # b1_word_ranks[b1] = rank_b1_list[-1]
                    baseline = baseline_list[-1]
                    print(t + " baseline rank: " + str(baseline))  # [0]
                    print_top_n_result(baseline_ranks, tokenizer_mlm)

                    t_id_ranks, t_cos_list = sort_vocab_with_log_freq(t_emb_new, output_embeddings, tokenizer_mlm,
                                                                      word_log_freq_dict, binary_freq=binary,
                                                                      norm_vec=True, square=square,
                                                                      norm_cos_sim_list=norm_cos, mean=mean)
                    parameters_t = parameters.copy()
                    parameters_t.append("category=" + category)
                    for s in show:
                        plot_annotate_top_n(w, t, t_id_ranks, t_cos_list, word_list=word_list, annotate_top_n=annotate,
                                            show_top_n=s, sort_cos_sim=sort_cos_sim, tokenizer=tokenizer_mlm,
                                            parameters=parameters_t)
                    target = w + ">" + t
                    word_ranks[target] = list(t_id_ranks).index(t_id) + 1
                    t_rank = word_ranks[target]
                    print(t + " rank: " + str(t_rank) + "(" + str(baseline - t_rank) + ")")  # [0]
                    print_top_n_result(t_id_ranks, tokenizer_mlm)
                    if baseline > t_rank:
                        t_win += 1
                category_word_ranks[category] = word_ranks
                category_output += category + " improve from baseline: " + str(t_win) + " out of " + str(
                    len(leave_one_out_list)) + "\n"
                print(category + " improve from baseline: " + str(t_win) + " out of " + str(len(leave_one_out_list)))
            print(category_output)
    print(parameters_b1)
    print(parameters_t)
    if run_word_pair:
        b1_output = ""
        b1_win = 0
        for b1, baseline, b1_rank in zip(b1_list, baseline_b1_list, rank_b1_list):
            # b1_output += (b1 + ": " + str(baseline) + "|" + str(b1_rank) + ",").rjust(15)
            b1_output += (b1 + ": " + str(b1_rank) + "(" + str(baseline - b1_rank) + ")" + ",").rjust(15)
            if baseline > b1_rank:
                b1_win += 1
        print(b1_output)
        print("b1 improve from baseline: " + str(b1_win) + " out of " + str(len(b1_list)))

        b2_output = ""
        b2_win = 0
        for b2, baseline, b2_rank in zip(b2_list, baseline_b2_list, rank_b2_list):
            # b2_output += (b2 + ": " + str(baseline) + "|" + str(b2_rank) + ",").rjust(15)
            b2_output += (b2 + ": " + str(b2_rank) + "(" + str(baseline - b2_rank) + ")" + ",").rjust(15)
            if baseline > b2_rank:
                b2_win += 1
        print(b2_output)
        print("b2 improve from baseline: " + str(b2_win) + " out of " + str(len(b2_list)))

    if run_category:
        for c in categories_word_list:
            print(c)
            print(dict(sorted(category_word_ranks[c].items(), key=lambda item: item[1])))


if __name__ == "__main__":
    main()
