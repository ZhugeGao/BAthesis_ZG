import glob
import math
import statistics

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from nltk.cluster import KMeansClusterer
import nltk
from transformers import BertTokenizer, BertForMaskedLM
import csv
import time
import numpy as np
from transformers import *

from numpy.linalg import norm

from collections import Counter

import argparse

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
    best = (np.argsort(np.dot(output_embeddings, embed.T), axis=0))
    # notes: dot product of all the embeddings and one embedding. rank it in ascending order.
    # The dot product of two normalized (unit) vectors will be a scalar value between -1 and 1.
    return list(best.T)


def sort_vocab_with_log_freq(embed, base_word_token, output_embeddings, tokenizer_mlm, word_log_freq_dict,
                             binary_freq=False,
                             norm_vec=True, square=True, norm_cos_sim_list=False, mean=None):
    assert mean in [None, 'geometric', 'harmonic'], 'mean could only be None, geometric or harmonic'

    if norm_vec:
        cos_sim_list = np.empty(len(output_embeddings))
        for i, emb in enumerate(output_embeddings):
            cos_sim_list[i] = np.dot(emb, embed.T) / (norm(emb) * norm(embed))
            if square:
                cos_sim_list[i] = np.sign(cos_sim_list[i]) * np.abs(cos_sim_list[i]) * np.abs(cos_sim_list[i])

    else:
        cos_sim_list = np.dot(output_embeddings, embed.T)  # transform to 0-1. normalize each pair

    # Separate lists for this as one example for the normalization part in parameter selection
    # print("min_cos_sim: ", np.min(cos_sim_list))
    # print("max_cos_sim: ", np.max(cos_sim_list))
    if norm_cos_sim_list:
        min_val = np.min(cos_sim_list)
        max_val = np.max(cos_sim_list)
        for i, cos_sim in enumerate(cos_sim_list):
            cos_sim_list[i] = (cos_sim - min_val) / (max_val - min_val)

        # print("min_cos_sim: (normalizing check)", np.min(cos_sim_list))
        # print("max_cos_sim: (normalizing check)", np.max(cos_sim_list))

    # print(cos_sim_list[0])
    # print(cos_sim_list[0].shape)
    # cnt = 0

    for i, cos_sim in enumerate(cos_sim_list):
        word_token = tokenizer_mlm.convert_ids_to_tokens(i)
        if (word_token in word_log_freq_dict) and (word_token != base_word_token):
            rel_log_freq = word_log_freq_dict[word_token]
            if binary_freq:
                rel_log_freq = 1.0
            # print("")
            # print(cos_sim_list[i])

            # geometric mean
            # print(cos_sim * rel_log_freq)
            if mean == "geometric":
                if norm_cos_sim_list:  # handle exception here?
                    cos_sim_list[i] = math.sqrt(
                        cos_sim * rel_log_freq)  # store the inverse log freq and use multiplication

            # harmonic mean
            if mean == "harmonic":
                cos_sim_list[i] = (cos_sim * rel_log_freq * 2.0) / (cos_sim + rel_log_freq)  # change into np.array
            if mean is None:
                cos_sim_list[i] = (cos_sim + rel_log_freq) / 2
            # print(cos_sim_list[i])

        else:
            # cos_sim_list[i] = np.divide(cos_sim, np.inf)
            cos_sim_list[i] = 0.0
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


def compute_log_frequency_for_lang(lang_word_list_path, base=10, sqrt=False):
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
            continue
        w2_id = tokenizer_mlm.encode(w2, add_special_tokens=False)
        w2_emb = output_embeddings[w2_id]
        if (len(w2_id) != 1):
            continue
        new_emb = w2_emb - w1_emb
        if w1 not in new_emb_dict:
            new_emb_dict[w1] = []
        new_emb_dict[w1].append(new_emb)
        # new_emb_list.append(new_emb)
    for k, v in new_emb_dict.items():
        new_emb_list.extend(np.average(v, axis=0))
    return np.average(new_emb_list, axis=0), new_emb_list


def print_top_n_result(rank, tokenizer_mlm, n=20):
    output = ""
    for result_id in list(rank)[:n - 1]:
        result_id = result_id.item()
        output += tokenizer_mlm.convert_ids_to_tokens(result_id) + ", "
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
        # print("number of " + str(len(word_list)) + " most frequent words in " + str(show_top_n) + ": " + str(
        #     len(highlights_x)))

        if len(highlights_x) > 0:
            print("mean: " + str(statistics.mean(highlights_x)))
            print("median: " + str(statistics.median(highlights_x)))
            # print("frequent word ranks: " + str(highlights_x))

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
        plt.annotate(text, xy=(x, cos_sim_dict[index]), fontsize=4)
    img_path = "../data/" + dir + '/' + title + '.svg'  # '.png'
    fig = plt.gcf()
    fig.savefig(img_path)
    plt.show()


def get_top_n_cos_sim(best, cos_sim_list, n=1000):
    new_cos_dict = {}
    for i in best[:n - 1]:
        new_cos_dict[i] = cos_sim_list[i]
    return new_cos_dict


def get_stats_for_list(l):
    copy = l.copy()
    stats = []
    stats.append(round(statistics.mean(copy), 3))
    stats.append(round(sum(i <= 1 for i in copy) / len(copy), 3))
    stats.append(round(sum(i <= 10 for i in copy) / len(copy), 3))
    stats.append(round(sum(i <= 50 for i in copy) / len(copy), 3))
    stats.append(round(sum(i <= 100 for i in copy) / len(copy), 3))

    return stats


def is_single_token_word(word):
    pretrained_weights = 'bert-base-uncased'  # -multilingual
    tokenizer_mlm = BertTokenizer.from_pretrained(pretrained_weights)
    id = tokenizer_mlm.encode(word, add_special_tokens=False)
    if len(id) != 1:
        return False


def get_all_diff_vecs(word_pair_list, tokenizer_mlm, output_embeddings):
    word_pair_category_list = []  # for inspection
    new_emb_list = []
    for w1, w2 in word_pair_list:
        w1_id = tokenizer_mlm.encode(w1, add_special_tokens=False)
        w1_emb = output_embeddings[w1_id]

        if len(w1_id) != 1:
            continue
        w2_id = tokenizer_mlm.encode(w2, add_special_tokens=False)
        w2_emb = output_embeddings[w2_id]
        if len(w2_id) != 1:
            continue
        diff_vec = w2_emb - w1_emb

        new_emb_list.append(np.array(diff_vec))  # .reshape()
        # new_emb_list.extend(np.array(diff_vec))
    return new_emb_list


def kmeans_cluster_diff_vectors(all_diff_vec_list, all_diff_vec_labels, number_of_categories, normalization=True,
                                distance="cosine"):
    assert normalization in [True, False], 'Normalization could only be True or False.'
    assert distance in ['cosine', 'Euclidean'], 'distance could only be cosine or Euclidean.'
    num_clusters = number_of_categories
    all_diff_vec_list = np.reshape(np.array(all_diff_vec_list), (len(all_diff_vec_list), 768))
    clusters_assignment = []
    if normalization:
        all_diff_vec_list = normalize(all_diff_vec_list)
    if distance == 'cosine':
        cluster_model = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters=True)
        clusters_assignment = cluster_model.cluster(all_diff_vec_list, assign_clusters=True)

    elif distance == 'Euclidean':
        cluster_model = KMeans(n_clusters=num_clusters, random_state=7)  # , algorithm="elkan", n_init=50
        cluster_model.fit(all_diff_vec_list)  # all the embeddings
        clusters_assignment = cluster_model.labels_

    clustered_vectors = [[] for i in range(num_clusters)]
    for vec_category_index, cl_id in enumerate(clusters_assignment):
        clustered_vectors[cl_id].append(all_diff_vec_labels[vec_category_index]) # give each diff vec a category and id

    category_total_counter = Counter([diff_vector for cluster in clustered_vectors for diff_vector in
                                      cluster])  # get a total counter. {cluster id: total_number_of_vector in cluster}
    # for each category, pick the main category: the majority located in which category.
    category_distribution_in_clusters = {}  # {category: Counter({cluster_id: number of category})}.
    # Get the main cluster automatically.(what if it is evenly distributed, then check the percentage)

    for category in category_total_counter.keys():
        if category not in category_distribution_in_clusters:
            category_distribution_in_clusters[category] = Counter()

    cluster_total_counter = {}
    for i, cluster in enumerate(
            clustered_vectors):  # loop over clusters. If it contain the category, add it to the category_distribution_in_clusters
        cluster_id = i + 1
        cluster_total_counter[cluster_id] = len(cluster)  # update cluster information in its total_counter
        # show the description of the cluster: what and how? In what format?
        cluster_counter = Counter(cluster)
        for category in category_distribution_in_clusters.keys():
            if category in cluster_counter:
                category_cluster_counter = category_distribution_in_clusters[category]
                dict.update(category_cluster_counter, [(cluster_id, cluster_counter[
                    category])])  # update the category distribution values to include data for this cluster

    all_main_clusters = []
    print(distance+ ","+ str(normalization))
    print("category,cluster_id,precision,recall,f_score")
    for category in sorted(category_distribution_in_clusters.keys()):  # loop over category
        category_total = category_total_counter[category]
        main_cluster_counter = category_distribution_in_clusters[category].most_common(1)

        # total number category and how much it is percentage
        # check for evenly distributed among clusters
        # get the data of main cluster
        main_cluster_id = main_cluster_counter[0][0]
        all_main_clusters.append(main_cluster_id)

        true_positive = main_cluster_counter[0][1]  # true positive: dominating one, first one. might be equal special cases
        false_positive = cluster_total_counter[main_cluster_id] - true_positive  # false positive: the remaining ones
        false_negative = category_total - true_positive  # false negative: the total of the dominating category minus the number of it in this.

        precision = round(true_positive / (true_positive + false_positive), 3) # Precision = TP/(TP + FP)
        recall = round(true_positive / (true_positive + false_negative), 3)  # Recall = TP / (TP + FN)
        f1_score = round(2 * (precision * recall)/(precision+recall), 3)

        print(category + "," + str(main_cluster_id)+"," + str(precision) +"," + str(recall) +"," + str(f1_score))


def main():
    run_category = True  # user arg.parser
    run_clustering = True

    show = [None]  # 10, 50, 100, 1000, 10000,
    show_plot = False
    sort_cos_sim = True

    # reorganize the parameters and Modularize

    mean_list = [None, 'geometric', 'harmonic']
    norm_cos_list = [True, False]
    binary_list = [True, False]
    square_list = [True, False]
    source_path = os.getcwd()
    data_path = source_path.replace("/source", "/data/")

    # load mBERT
    # make an argument to switch between bert and multilingual bert
    pretrained_weights = 'bert-base-uncased'  # -multilingual
    tokenizer_mlm = BertTokenizer.from_pretrained(pretrained_weights)
    model_mlm = BertForMaskedLM.from_pretrained(pretrained_weights, output_hidden_states=True)
    output_embeddings = model_mlm.cls.predictions.decoder.weight.detach().cpu().numpy()

    annotate = 50
    highlight = 200
    word_log_freq_dict = compute_log_frequency_for_lang(data_path + "en/en_50k.txt", sqrt=False)
    word_list = list(word_log_freq_dict.keys())
    word_list = word_list[:highlight]

    if run_clustering:
        os.chdir(source_path.replace("/source", "/data/word_lists_single_token/"))
        all_diff_vec_list = []
        all_diff_vec_labels = []
        # maybe a new method read all word list

        category_list = glob.glob("*.csv")
        number_of_category = len(category_list)
        # normalize embeddings

        for word_list_path in category_list:
            category = word_list_path
            word_pair_list = []
            with open(word_list_path) as words:
                reader = csv.reader(words)
                for row in reader:
                    w1, w2 = row
                    word_pair_list.append([w1, w2])

            category_diff_vec_list = get_all_diff_vecs(word_pair_list, tokenizer_mlm, output_embeddings)
            # collect all the difference vector

            category_diff_vec_labels = [category] * len(category_diff_vec_list)

            all_diff_vec_list.extend(category_diff_vec_list)
            all_diff_vec_labels.extend(category_diff_vec_labels)

        distances = ['Euclidean', "cosine"]
        normalizations = [True, False]
        for distance in distances:
            for normalization in normalizations:
                kmeans_cluster_diff_vectors(all_diff_vec_list, all_diff_vec_labels, number_of_category, distance=distance, normalization=normalization)

    mean = 'geometric'
    norm_cos = True
    binary = False
    square = True

    if run_category:
        # run category binary switch
        parameters = ["mean=" + str(mean), "norm_cos=" + str(norm_cos), "binary=" + str(binary),
                      "square=" + str(square)]
        data_dir = data_path
        for p in parameters:
            data_dir += p + ","
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        category_word_ranks = {}
        category_word_baselines = {}
        category_win_dict = {}
        if run_category:
            category_output = ""
            os.chdir(source_path.replace("/source", "/data/word_lists_single_token/"))

            for word_list_path in glob.glob("*"):
                t_total = 0
                t_hard_win = 0
                t_soft_win = 0
                word_ranks = {}
                # word_baseline = {}
                word_pair_list = []
                baseline_list = []
                with open(word_list_path) as words:  # {}.csv".format(category)
                    category = word_list_path
                    reader = csv.reader(words)
                    for row in reader:
                        w1, w2 = row
                        word_pair_list.append([w1, w2])

                # leave one out: using a for loop
                leave_one_out_list = [[e for e in word_pair_list if e != word_pair_list[i]] for i in
                                      range(len(word_pair_list))]

                for i, l in enumerate(leave_one_out_list):
                    w = word_pair_list[i][0].lower()
                    w_id = tokenizer_mlm.encode(w, add_special_tokens=False)
                    if len(w_id) != 1: continue
                    w_emb = output_embeddings[w_id]

                    t = word_pair_list[i][1].lower()
                    t_id = tokenizer_mlm.encode(t, add_special_tokens=False)
                    if len(t_id) != 1: continue

                    avg_vec, category_diff_vec_list = get_average_vec_for_category(w, l, tokenizer_mlm,
                                                                                   output_embeddings)

                    t_emb_new = w_emb + avg_vec
                    # print(w, t)
                    baseline_ranks, baseline_cos_list = sort_vocab_with_log_freq(w_emb, w, output_embeddings,
                                                                                 tokenizer_mlm,
                                                                                 word_log_freq_dict,
                                                                                 binary_freq=binary, norm_vec=True,
                                                                                 square=square,
                                                                                 norm_cos_sim_list=norm_cos,
                                                                                 mean=mean)
                    baseline_list.append(list(baseline_ranks).index(t_id) + 1)

                    baseline = baseline_list[-1]

                    t_id_ranks, t_cos_list = sort_vocab_with_log_freq(t_emb_new, w, output_embeddings,
                                                                      tokenizer_mlm,
                                                                      word_log_freq_dict, binary_freq=binary,
                                                                      norm_vec=True, square=square,
                                                                      norm_cos_sim_list=norm_cos, mean=mean)
                    parameters_t = parameters.copy()
                    # former plot
                    target = w + ">" + t
                    t_rank = list(t_id_ranks).index(t_id) + 1

                    word_ranks[target] = str(t_rank) + "(" + str(baseline - t_rank) + ")"
                    # print(t + " baseline rank: " + str(baseline))  # [0]
                    # print_top_n_result(baseline_ranks, tokenizer_mlm)
                    # print(t + " rank: " + str(t_rank) + "(" + str(baseline - t_rank) + ")")  # [0]
                    # print_top_n_result(t_id_ranks, tokenizer_mlm)
                    if baseline >= t_rank:
                        t_soft_win += 1
                    if baseline > t_rank:
                        t_hard_win += 1
                    t_total += 1
                    if show_plot:
                        for s in show:
                            plot_annotate_top_n(w, t, baseline_ranks, baseline_cos_list, word_list=word_list,
                                                annotate_top_n=annotate,
                                                show_top_n=s, sort_cos_sim=sort_cos_sim, tokenizer=tokenizer_mlm,
                                                parameters=parameters_t)
                            plot_annotate_top_n(w, t, t_id_ranks, t_cos_list, word_list=word_list,
                                                annotate_top_n=annotate,
                                                show_top_n=s, sort_cos_sim=sort_cos_sim, tokenizer=tokenizer_mlm,
                                                parameters=parameters_t)
                category_word_ranks[category] = word_ranks
                category_output += category + " improve from baseline: " + str(t_hard_win) + " out of " + str(
                    t_total) + "\n"
                # print(category + " improve from baseline: " + str(t_hard_win) + " out of " + str(t_total))
                # print(category + "," + str(round(t_hard_win / t_total, 3)) + "," + str(
                #     round(t_soft_win / t_total, 3)))
                category_win_dict[category] = [round(t_hard_win / t_total, 3), round(t_soft_win / t_total, 3)]
            # print("")
            # print(category_output)
            category_file_list = glob.glob("*")  # for code readability
            print("")
            print("category,avg_baseline_rank,avg_rank,acc@1,acc@10,acc@50,acc@100,hard_win,soft_win")
            for c in category_file_list:
                ranks = []
                improve_from_baselines = []
                baseline = []
                for v in category_word_ranks[c].values():
                    ranks.append(int(v[:v.index("(")]))
                    improve_from_baselines.append(int(v[v.index("(") + 1:v.index(")")]))
                    baseline.append(ranks[-1] + improve_from_baselines[-1])  # optimize

                ranks_stats = get_stats_for_list(ranks)
                ranks_stats.insert(0, round(statistics.mean(baseline), 3))
                ranks_stats.extend(category_win_dict[c])

                print(c + "," + str(ranks_stats)[1:-1])
            print("")
            print("category, base>derivate : rank(improve_from_baseline)")
            for c in category_file_list:
                print(c + "," + str(category_word_ranks[c])[1:-1].replace("'",""))
                # print(c + "," + str(sorted(category_word_ranks[c].items(), key=lambda item: item[1])))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
