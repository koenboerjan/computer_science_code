import numpy as np
import kshingle as ks
import re
from cosine import get_cosine_result
from data import extract_model_words


def optimize_epsilon(potential_matches_block: list[list[int]], all_data: list[dict],
                     known_brands: set[str], real_pairs) -> float:
    """
    Optimize epsilon for given bootstrap sample with highest F1 score.

    :param potential_matches_block:
    :param all_data:
    :param known_brands:
    :param real_pairs:
    :return:
    """
    f1_base = 0
    improvement = 0
    epsilon = 0.4
    F1_result = []
    while improvement < 2:
        clusters = perform_msm(potential_matches_block, all_data, known_brands, epsilon)
        pq, pc, f1, frac = evaluate_msm(clusters, real_pairs)
        print(epsilon)
        print(f"PQ: {pq}, PC: {pc}, F1: {f1}, frac: {frac}")
        F1_result.append(f1)
        if f1 > f1_base:
            improvement = 0
            f1_base = f1
        else:
            improvement += 1
        epsilon += 0.02
    print(F1_result)
    return epsilon


def perform_msm(potential_matches_block: list[list[int]], all_data: list[dict],
                known_brands: set[str], epsilon) -> list[set[int]]:
    matched_pairs = []
    data_size = len(all_data)
    similarity_matrix = np.eye(data_size)
    for block in potential_matches_block:
        # print(block)
        similarity_matrix = perform_msm_per_block(block, all_data, similarity_matrix, known_brands)
        # Clustering
        # n_items = len(block)
        # epsilon = 0.5
        # for x1 in range(0, n_items):
        #     for x2 in range(x1 + 1, n_items):
        #         if similarity_matrix[x1, x2] > epsilon:
        #             if not(matched_pairs in matched_pairs):
        #                 matched_pairs.append({block[x1],block[x2]})
    similarity_matrix = similarity_matrix - np.eye(data_size)
    print("letsgo Clustering")
    # print([max(similarity_matrix[i]) for i in range(0,10)])
    clusters = hierarchical_clustering(similarity_matrix, epsilon)

    return clusters


def recursive_search(similarity_matrix, current_cluster, last_cluster=10000) -> (int, int):
    # new_cluster = similarity_matrix[current_cluster].index(max(similarity_matrix[current_cluster]))
    new_cluster = [value for value in range(0, len(similarity_matrix[current_cluster])) if
                   (similarity_matrix[current_cluster, value] == max(similarity_matrix[current_cluster]))][0]
    if new_cluster == last_cluster:
        return current_cluster, last_cluster
    else:
        return (recursive_search(similarity_matrix=similarity_matrix,
                                 current_cluster=new_cluster,
                                 last_cluster=current_cluster))


def hierarchical_clustering(similarity_matrix, epsilon) -> list[set[int]]:
    clusters = [[index] for index in range(0, len(similarity_matrix))]
    start = 0
    unfinished = True
    while unfinished:
        while max(similarity_matrix[start]) < epsilon and start < len(similarity_matrix) - 1:
            # print(max(similarity_matrix[start]))
            start += 1
            # print(start)
        # print(max(similarity_matrix[start]))
        cluster_1, cluster_2 = recursive_search(similarity_matrix, start)
        if similarity_matrix[cluster_1, cluster_2] > epsilon:
            clusters[cluster_1].extend(clusters[cluster_2])
            clusters.remove(clusters[cluster_2])
            similarity_matrix[cluster_1] = np.array(
                [(similarity_matrix[cluster_1, i] + similarity_matrix[cluster_2, i])/2 for
                 i in range(0, len(similarity_matrix[cluster_1]))])
            similarity_matrix = np.delete(similarity_matrix, cluster_2, axis=0)

            for i in range(0, len(similarity_matrix)):
                similarity_matrix[i][cluster_1] = (similarity_matrix[i, cluster_1] + similarity_matrix[i, cluster_2])/2

            similarity_matrix = np.delete(similarity_matrix, cluster_2, axis=1)
        else:
            start += 1
        if start == len(similarity_matrix):
            unfinished = False

    print(clusters)
    print(len(clusters))
    print([cluster for cluster in clusters if len(cluster) > 1])
    print([cluster for cluster in clusters if len(cluster) > 2])

    return [set(cluster) for cluster in clusters if len(cluster) > 1]


def evaluate_msm(clustered_pairs, real_pairs):
    found_comparisons = 0
    missing_comparison = 0
    print(clustered_pairs)
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
    # print(len(one_on_one_real_pairs))
    # print(found_comparisons)
    # print(missing_comparison)
    frac_comp = count_comp / (1624 * 1623 / 2)
    if frac_comp > 1:
        frac_comp = 1
    print(f"fraction of comparisons: {count_comp / (1624 * 1623 / 2)}")
    PQ = found_comparisons / count_comp
    print(f"PQ: {PQ}")
    PC = found_comparisons / len(real_pairs)
    print(f"PC: {PC}")
    F_star = 2 * PC * PQ / (PC + PQ)
    print(f"F1*: {F_star}")
    return PQ, PC, F_star, frac_comp


def check_for_same_brand(item_1, item_2, known_brands):
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


def check_for_same_shop(item_1, item_2):
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


def calc_sim(shingles_1, shingles_2):
    n1 = len(shingles_1)
    n2 = len(shingles_2)
    same_shingles = shingles_1 & shingles_2
    different_shingles = shingles_1.difference(same_shingles) | shingles_2.difference(same_shingles)
    if (n1 + n2) > 0:
        q_gram_sim = (n1 + n2 - len(different_shingles)) / (n1 + n2)
    else:
        q_gram_sim = 0
    return q_gram_sim


def key_value_pair_comparison(item1_dict, item2_dict):
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
            q_gram_sim = calc_sim(shingles_item1[x1], shingles_item2[x2])
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
                    similarity += calc_sim(value1_shingles, value2_shingles) * q_gram_sim
                    m += 1
                    # print(similarity)
    min_features = min(len(item1_dict), len(item2_dict))
    return similarity, m / min_features


def jaccard_sim(set1, set2):
    return len(set1 & set2) / len(set1 | set2)


def perform_msm_per_block(product_block: list[int], all_data: list[dict], old_sim_matrix, known_brands: set[str]):
    n_items = len(product_block)
    mu = 0.65
    new_sim_matrix = old_sim_matrix
    for i in range(0, n_items):
        for j in range(i + 1, n_items):
            x1 = product_block[i]
            x2 = product_block[j]
            data_i = all_data[x1]
            data_j = all_data[x2]
            if (not (check_for_same_brand(data_i, data_j, known_brands)) or
                    data_i['shop'] == data_j['shop']):
                new_sim_matrix[x1, x2] = -10
                new_sim_matrix[x2, x1] = -10
            if new_sim_matrix[x2, x1] == 0:
                similarity, m_weight = key_value_pair_comparison(data_i['featuresMap'], data_j['featuresMap'])
                theta_1 = (1 - mu) * m_weight
                theta_2 = 1 - mu - theta_1
                similarity = similarity * theta_1
                similarity += mu * get_cosine_result(data_i['title'], data_j['title'])
                all_model_words, sets_of_words = extract_model_words([data_i, data_j], known_brands,
                                                                     include_feature=True)
                similarity += theta_2 * jaccard_sim(sets_of_words[0], sets_of_words[1])
                new_sim_matrix[x2, x1] = similarity
                new_sim_matrix[x1, x2] = similarity
    return new_sim_matrix
