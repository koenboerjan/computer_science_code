import numpy as np
import kshingle as ks
import re
from cosine import get_cosine_result
from data import extract_model_words
import typing


def optimize_epsilon(potential_matches_block: list[list[int]], all_data: list[dict[str | dict[str]]],
                     known_brands: set[str], real_pairs: list[set[int]]) -> float:
    """
    Optimize epsilon for given bootstrap sample with highest F1 score. Start with epsilon = 0.52 and decrease till
    optimum reached

    :param potential_matches_block: list of all blocks of candidate pairs
    :param all_data: all data items
    :param known_brands: set of all known brand
    :param real_pairs: real duplicate pairs
    :return: the optimal epsilon value
    """
    f1_base = 0
    no_improvement = 0
    epsilon = 0.52
    best_epsilon = epsilon
    f1_result = []

    data_size = len(all_data)
    similarity_matrix = np.eye(data_size)
    for block in potential_matches_block:
        similarity_matrix = perform_msm_per_block(product_block=block,
                                                  all_data=all_data,
                                                  old_sim_matrix=similarity_matrix,
                                                  known_brands=known_brands)
    similarity_matrix = similarity_matrix - np.eye(data_size)
    clusters = [[index] for index in range(0, len(similarity_matrix))]

    while no_improvement < 2:
        clusters, similarity_matrix = hierarchical_clustering(similarity_matrix=similarity_matrix,
                                                              epsilon=epsilon,
                                                              clusters=clusters)
        cluster_set = [set(cluster) for cluster in clusters if len(cluster) > 1]
        pq, pc, f1, frac = evaluate_msm(clustered_pairs=cluster_set,
                                        real_pairs=real_pairs,
                                        len_data=len(all_data))
        f1_result.append(f1)
        if f1 > f1_base:
            no_improvement = 0
            f1_base = f1
            best_epsilon = epsilon
        else:
            no_improvement += 1
        epsilon -= 0.02
    return best_epsilon


def perform_msm(potential_matches_block: list[list[int]], all_data: list[dict[str | dict[str]]],
                known_brands: set[str], epsilon: float) -> list[set[int]]:
    """
    Performs msm over the full dataset with all blocks of candidate pairs.

    :param potential_matches_block: list of all blocks of candidate pairs
    :param all_data: complete dataset
    :param known_brands: set of all known brands
    :param epsilon: epsilon used for clustering
    :return: list of clusters of duplicates
    """
    data_size = len(all_data)
    similarity_matrix = np.eye(data_size)
    for block in potential_matches_block:
        similarity_matrix = perform_msm_per_block(block, all_data, similarity_matrix, known_brands)
    similarity_matrix = similarity_matrix - np.eye(data_size)

    print("Lets go Clustering")
    clusters_base = [[index] for index in range(0, len(similarity_matrix))]
    clusters, similarity_matrix = hierarchical_clustering(similarity_matrix=similarity_matrix,
                                                          epsilon=epsilon,
                                                          clusters=clusters_base)

    return [set(cluster) for cluster in clusters if len(cluster) > 1]


def clustering_search(similarity_matrix: typing.Any) -> (int, int):
    """
    Retrieve the position in the similarity matrix which has the biggest similarity

    :param similarity_matrix: similarity matrix
    :return: location of highest similarity
    """
    max_sim = 0
    cluster_1 = 0
    cluster_2 = 0
    for x1 in range(0, len(similarity_matrix)):
        for x2 in range(0, len(similarity_matrix)):
            if max_sim < similarity_matrix[x1, x2]:
                max_sim = similarity_matrix[x1, x2]
                cluster_1 = x1
                cluster_2 = x2
    return cluster_1, cluster_2


def hierarchical_clustering(similarity_matrix: typing.Any, epsilon: float, clusters: list[list[int]]) -> \
        (list[list[int]], list[list[int]]):
    """
    Perform hierarchical clustering based on similarity matrix and epsilon.

    :param similarity_matrix: similarity matrix of potential pairs.
    :param epsilon: threshold value for clustering
    :param clusters: clusters which have already been determined.
    :return: list of cluster sets and updated similarity matrix
    """
    unfinished = True
    while unfinished:
        cluster_1, cluster_2 = clustering_search(similarity_matrix=similarity_matrix)
        if similarity_matrix[cluster_1, cluster_2] > epsilon:
            clusters[cluster_1].extend(clusters[cluster_2])
            clusters.remove(clusters[cluster_2])
            similarity_matrix[cluster_1] = np.array(
                [min(similarity_matrix[cluster_1, i], similarity_matrix[cluster_2, i]) for
                 i in range(0, len(similarity_matrix[cluster_1]))])
            similarity_matrix = np.delete(similarity_matrix, cluster_2, axis=0)

            for i in range(0, len(similarity_matrix)):
                similarity_matrix[i][cluster_1] = min(similarity_matrix[i, cluster_1],
                                                      similarity_matrix[i, cluster_2])

            similarity_matrix = np.delete(similarity_matrix, cluster_2, axis=1)
        else:
            unfinished = False
    return clusters, similarity_matrix


def evaluate_msm(clustered_pairs: list[set[int]], real_pairs: list[set[int]], len_data: int) -> \
        (float, float, float, float):
    """
    Evaluate msm based on pc, pq and f1, versus fraction of comparisons

    :param clustered_pairs: all pairs selected as duplicates
    :param real_pairs: all real duplicates
    :param len_data: length of dataset
    :return: pq, pc, f1 and fraction
    """
    found_comparisons = 0
    missing_comparison = 0
    for real_pair in real_pairs:
        found = False
        for comparison in clustered_pairs:
            if not (real_pair.difference(set(comparison)) & real_pair):
                found = True
                break
        if found:
            found_comparisons += 1
        else:
            missing_comparison += 1

    count_comp = 0
    for comparison in clustered_pairs:
        count_comp += len(comparison) * (len(comparison) - 1) / 2
    frac_comp = count_comp / (len_data * (len_data - 1) / 2)
    if frac_comp > 1:
        frac_comp = 1
    pq = found_comparisons / count_comp
    pc = found_comparisons / len(real_pairs)
    if (pc + pq) != 0:
        f_star = 2 * pc * pq / (pc + pq)
    else:
        f_star = 0
    return pq, pc, f_star, frac_comp


def check_for_same_brand(item_1: dict[str | dict[str]], item_2: dict[str | dict[str]], known_brands: set[str]) \
        -> set[str] | bool:
    """
    Check if two items have the same brand based on known set of brands.

    :param item_1: first item to check
    :param item_2: second item to check
    :param known_brands: all brands that are known
    :return: set of brands, can be used as boolean
    """
    brand_found = []
    i = 0
    for item in [item_1, item_2]:
        item_set = set(item['title'].lower().split())
        try:
            item_set.add(item['featuresMap']['Brand'])
        except:
            pass
        brand_found.append(item_set & known_brands)
        i += 1
    if not (brand_found[0] and brand_found[1]):
        return True
    else:
        return brand_found[0] & brand_found[1]


def calc_sim(shingles_1: set[str], shingles_2: set[str]) -> float:
    """
    Calculate q-gram similarity between two set of shingles

    :param shingles_1: set of shingles item 1
    :param shingles_2: set of shingles item 2
    :return: q-gram similarity between two sets of shingles
    """
    n1 = len(shingles_1)
    n2 = len(shingles_2)
    same_shingles = shingles_1 & shingles_2
    different_shingles = shingles_1.difference(same_shingles) | shingles_2.difference(same_shingles)
    if (n1 + n2) > 0:
        q_gram_sim = (n1 + n2 - len(different_shingles)) / (n1 + n2)
    else:
        q_gram_sim = 0
    return q_gram_sim


def key_value_pair_comparison(item1_dict: dict, item2_dict: dict) -> (float, float):
    """
    Calculate key value pairs similarity for 2 items.
    if Key values are of different data type(one numerical, other string), similarity again set to zero.

    :param item1_dict: dict filled with features of item 1
    :param item2_dict: dict filled with features of item 1
    :return: similarity and percentage of features that are taken into account
    """
    q = 3
    regex_numerical = re.compile('[0-9]+')
    shingles_item1 = [set(ks.shingleseqs_list(key_item1.lower(), klist=[q])[0]) for key_item1 in item1_dict.keys()]
    shingles_item2 = [set(ks.shingleseqs_list(key_item2.lower(), klist=[q])[0]) for key_item2 in item2_dict.keys()]
    p1 = len(shingles_item1)
    p2 = len(shingles_item2)
    similarity = 0
    m = 0
    for x1 in range(0, p1):
        for x2 in range(0, p2):
            q_gram_sim = calc_sim(shingles_1=shingles_item1[x1],
                                  shingles_2=shingles_item2[x2])
            if q_gram_sim > 0.75:
                key_item1 = [key_item1 for key_item1 in item1_dict.keys()][x1]
                key_item2 = [key_item2 for key_item2 in item2_dict.keys()][x2]
                value1 = item1_dict[key_item1]
                value2 = item2_dict[key_item2]
                numerical1 = set(filter(regex_numerical.match, value1))
                numerical2 = set(filter(regex_numerical.match, value2))
                if (numerical1 and numerical2) or (not numerical1 and not numerical2):
                    value1_shingles = set(ks.shingleseqs_list(value1.lower(), klist=[2])[0])
                    value2_shingles = set(ks.shingleseqs_list(value2.lower(), klist=[2])[0])
                    similarity += calc_sim(shingles_1=value1_shingles,
                                           shingles_2=value2_shingles) * q_gram_sim
                    m += 1
    min_features = min(len(item1_dict), len(item2_dict))
    return similarity, m / min_features


def jaccard_sim(set1: set[str], set2: set[str]) -> float:
    return len(set1 & set2) / len(set1 | set2)


def perform_msm_per_block(product_block: list[int], all_data: list[dict[str | dict[str]]], old_sim_matrix: typing.Any,
                          known_brands: set[str]) -> typing.Any:
    """
    Perform MSM for a block of candidate pairs, first check if two products have different brand or are in the same shop,
    if this is the case their distance is put to negative and ignored for further computation.
    First step after that is calculating key_value similarity
    Second step is performing MSM by calculating the cosine similarity and jaccard similarity between model words.
    Also model words found in key_value pairs are now included.
    All these values are combined by weight.


    :param product_block: block of candidate pairs.
    :param all_data: all data considered in the global state
    :param old_sim_matrix: similarity matrix that will be updated
    :param known_brands: brands
    :return: updated similarity matrix with values for pairs present in block
    """
    n_items = len(product_block)
    mu = 0.65
    new_sim_matrix = old_sim_matrix
    for i in range(0, n_items):
        for j in range(i + 1, n_items):
            x1 = product_block[i]
            x2 = product_block[j]
            data_i = all_data[x1]
            data_j = all_data[x2]
            if (not (check_for_same_brand(item_1=data_i, item_2=data_j, known_brands=known_brands)) or
                    data_i['shop'] == data_j['shop']):
                new_sim_matrix[x1, x2] = -10
                new_sim_matrix[x2, x1] = -10
            if new_sim_matrix[x2, x1] == 0:
                similarity, m_weight = key_value_pair_comparison(item1_dict=data_i['featuresMap'],
                                                                 item2_dict=data_j['featuresMap'])
                theta_1 = (1 - mu) * m_weight
                theta_2 = 1 - mu - theta_1
                similarity = similarity * theta_1
                similarity += mu * get_cosine_result(content_a=data_i['title'],
                                                     content_b=data_j['title'])
                all_model_words, sets_of_words = extract_model_words(data=[data_i, data_j],
                                                                     known_brands=known_brands,
                                                                     include_feature=True)
                similarity += theta_2 * jaccard_sim(set1=sets_of_words[0],
                                                    set2=sets_of_words[1])
                new_sim_matrix[x2, x1] = similarity
                new_sim_matrix[x1, x2] = similarity
    return new_sim_matrix
