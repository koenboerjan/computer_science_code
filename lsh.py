import math
import matplotlib.pyplot as plt
from data import determine_real_pairs


def minhash_function(a, b, max_col):
    buckets = 2000
    return [(a * x + b) % buckets for x in range(1, max_col)]


def create_signature_matrix(all_model_words: set[str], model_words_per_tv: list[set[str]]):
    length_minhash = len(all_model_words)
    a = [11, 13, 15, 17, 19, 23, 29, 31, 37, 41]
    # 23, 29, 31, 37, 41
    b = [1, 5, 3, 6, 2, 7]
    min_hashes = []
    for i in range(0, len(b)):
        for j in range(0, len(a)):
            min_hashes.append(minhash_function(a=a[j],
                                               b=b[i],
                                               max_col=length_minhash + 1))

    sig_matrix = []
    minhash_n = len(a) * len(b)
    # minhash_n = 30
    # shuffle_array = []
    # min_hashes = []
    # for a in range(0,length_minhash):
    #     shuffle_array.append(a)
    # for s in range(0, length_minhash):
    #     min_hashes.append(mit.random_permutation(shuffle_array))

    for tv in model_words_per_tv:
        binary_vec = [word in tv for word in all_model_words]
        signature_col = [min([binary_vec[i] * min_hashes[j][i] for i in range(0, len(binary_vec)) if binary_vec[i]]) for
                         j in range(0, minhash_n)]
        sig_matrix.append(signature_col)
    return sig_matrix


def hash_signature(sig_values: list[int], band: int) -> int:
    power_val = len(sig_values)
    bucket = band * 10 ** ((power_val + 1) * 3)
    for i in range(0, power_val):
        bucket += sig_values[i] * 10 ** ((power_val - i) * 3)
    return bucket


def return_potential_matches(signature_matrix: list[list[int]], bands: int):
    n_rows = len(signature_matrix[1])
    band_size = math.floor(n_rows / bands)
    if n_rows % bands != 0:
        raise RuntimeError("rows and bands not dividable")
    potential_matches = []
    for i in range(0, bands):
        band_max = (i + 1) * band_size
        band_min = i * band_size
        buckets_with_matches = set()
        buckets_sig_hash = []

        for tv_sig in signature_matrix:
            bucket_sig_hash = hash_signature(sig_values=tv_sig[band_min: band_max], band=i)
            if bucket_sig_hash in buckets_sig_hash:
                buckets_with_matches.add(bucket_sig_hash)
            buckets_sig_hash.append(bucket_sig_hash)

        for match in buckets_with_matches:
            matching_tv = [tv_index for tv_index in range(0, len(buckets_sig_hash) - 1) if
                           buckets_sig_hash[tv_index] == match]
            potential_matches.append(matching_tv)

    # return_groups = set()
    # count = 0
    # count_comp = 0
    # for buckets_with_matches in potential_matches:
    #     if set(buckets_with_matches) & return_groups:
    #         count += 1
    #     count_comp += len(buckets_with_matches) * (len(buckets_with_matches) - 1) / 2
    #     return_groups = return_groups | set(buckets_with_matches)
    # print(count)
    # print(count_comp)
    return potential_matches


def evaluate_lsh(real_pairs: list[set[int]], evaluate_blocks: list[list[int]]):
    found_comparisons = 0
    missing_comparison = 0

    for real_pair in real_pairs:
        found = False
        for comparison in evaluate_blocks:
            if not (real_pair.difference(set(comparison)) & real_pair):
                found = True
                # print(comparison)
                # print(real_pair)
                break
        if found:
            found_comparisons += 1
        else:
            missing_comparison += 1

    count_comp = 0
    for comparison in evaluate_blocks:
        count_comp += len(comparison) * (len(comparison) - 1) / 2
    # print(len(one_on_one_real_pairs))
    # print(found_comparisons)
    # print(missing_comparison)
    frac_comp = count_comp / (1624 * 1623 / 2)
    if frac_comp > 1:
        frac_comp = 1
    # print(f"fraction of comparisons: {count_comp/(1624*1623/2)}")
    PQ = found_comparisons / count_comp
    # print(f"PQ: {PQ}")
    PC = found_comparisons / len(real_pairs)
    # print(f"PC: {PC}")
    F_star = 2 * PC * PQ / (PC + PQ)
    # print(f"F1*: {F_star}")
    return PQ, PC, F_star, frac_comp


def generate_lsh_plots(signature_matrix, all_tv):
    test_bands = [2, 3, 4, 5, 6, 10, 12, 15, 20, 30]
    pq_plot = []
    pc_plot = []
    f1_plot = []
    frac_plot = []
    for test_band in test_bands:
        pq_mean = 0
        pc_mean = 0
        f1_mean = 0
        frac_comp_mean = 0
        iterations = 5
        real_pairs = determine_real_pairs(all_tv)
        for i in range(0, iterations):
            matches = return_potential_matches(bands=test_band,
                                               signature_matrix=signature_matrix)
            run_pq, run_pc, run_f1, run_frac = evaluate_lsh(real_pairs, matches)
            pq_mean += 1 / iterations * run_pq
            pc_mean += 1 / iterations * run_pc
            f1_mean += 1 / iterations * run_f1
            frac_comp_mean += 1 / iterations * run_frac
        pq_plot.append(pq_mean)
        pc_plot.append(pc_mean)
        f1_plot.append(f1_mean)
        frac_plot.append(frac_comp_mean)
    plt.plot(frac_plot, pq_plot)
    plt.show()
    plt.plot(frac_plot, pc_plot)
    plt.show()
    plt.plot(frac_plot, f1_plot)
    plt.show()
    print(f"fraction: {frac_plot} \n"
          f"pq: {pq_plot} \n"
          f"pc: {pc_plot} \n"
          f"f1: {f1_plot}")
