import math
import matplotlib.pyplot as plt
from data import determine_real_duplicates
# import more_itertools as mit


def minhash_function(a, b, max_col) -> list[int]:
    """
    minhash function for pseudo random signature matrix creation.

    :param a: parameter in function
    :param b: parameter in function
    :param max_col: length of binary vector
    :return: vector of minhash values of length of binary vector
    """
    buckets = 2000
    return [(a * x + b) % buckets for x in range(1, max_col)]


def create_signature_matrix(all_model_words: set[str], model_words_per_tv: list[set[str]]) -> list[list[int]]:
    """
    Create a signature matrix for all different tvs, based on model words found for every tv.

    :param all_model_words: total set of model words
    :param model_words_per_tv: set of model words per tv
    :return: signature matrix
    """
    length_minhash = len(all_model_words)
    a = [11, 13, 15, 17, 19, 23, 29, 31, 37, 41]
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
    """
    Hashes signature to specific bucket.

    :param sig_values: values of signature in specific band
    :param band: index of band
    :return: bucket_id
    """
    power_val = len(sig_values)
    bucket = band * 10 ** ((power_val + 1) * 3)
    for i in range(0, power_val):
        bucket += sig_values[i] * 10 ** ((power_val - i) * 3)
    return bucket


def return_candidate_pairs(signature_matrix: list[list[int]], bands: int) -> list[list[int]]:
    """
    Return potential matches based on if they hash bands to the same bucket.

    :param signature_matrix: signature matrix of observations
    :param bands: total count of bands
    :return: list of list of tvs which hash to the same bucket and are candidate pairs.
    """
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
            bucket_sig_hash = hash_signature(sig_values=tv_sig[band_min: band_max],
                                             band=i)
            if bucket_sig_hash in buckets_sig_hash:
                buckets_with_matches.add(bucket_sig_hash)
            buckets_sig_hash.append(bucket_sig_hash)

        for match in buckets_with_matches:
            matching_tv = [tv_index for tv_index in range(0, len(buckets_sig_hash) - 1) if
                               buckets_sig_hash[tv_index] == match]
            potential_matches.append(matching_tv)
    return potential_matches


def evaluate_lsh(real_duplicates: list[set[int]], evaluate_blocks: list[list[int]], observations_count: int) -> (float, float, float, float):
    """
    Compute evaluation values for lsh

    :param observations_count: amount of observations which are evaluated
    :param real_duplicates: real pairs within dataset
    :param evaluate_blocks: blocks of candidate pairs determined by lsh
    :return: different scores
    """
    found_duplicate = 0
    missing_duplicates = 0

    for duplicate in real_duplicates:
        found = False
        for comparison in evaluate_blocks:
            if not (duplicate.difference(set(comparison)) & duplicate):
                found = True
                break
        if found:
            found_duplicate += 1
        else:
            missing_duplicates += 1

    count_comp = 0
    for comparison in evaluate_blocks:
        count_comp += len(comparison) * (len(comparison) - 1) / 2
    frac_comp = count_comp / (observations_count * (observations_count - 1) / 2)

    #Correct for comparison more than 1, then using lsh does not have positive effect.
    if frac_comp > 1:
        frac_comp = 1

    # print(f"fraction of comparisons: {count_comp/(1624*1623/2)}")
    pq = found_duplicate / count_comp
    # print(f"pq: {pq}")
    pc = found_duplicate / len(real_duplicates)
    # print(f"pc: {pc}")
    f_star = 2 * pc * pq / (pc + pq)
    # print(f"F1*: {f_star}")
    return pq, pc, f_star, frac_comp


def generate_lsh_summary_and_plots(signature_matrix: list[list[int]], all_tv: list[dict]):
    """
    Generate summary data and performance for different band sizes of LSH, iterating 5 times as model words set
    has random order.

    :param signature_matrix: signature matrix created by lsh
    :param all_tv: all tv dataset
    :return: plots and summary
    """
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
        real_pairs = determine_real_duplicates(all_tv)
        for i in range(0, iterations):
            matches = return_candidate_pairs(bands=test_band,
                                             signature_matrix=signature_matrix)
            run_pq, run_pc, run_f1, run_frac = evaluate_lsh(real_duplicates=real_pairs,
                                                            evaluate_blocks=matches,
                                                            observations_count=len(all_tv))
            pq_mean += 1 / iterations * run_pq
            pc_mean += 1 / iterations * run_pc
            f1_mean += 1 / iterations * run_f1
            frac_comp_mean += 1 / iterations * run_frac
        pq_plot.append(pq_mean)
        pc_plot.append(pc_mean)
        f1_plot.append(f1_mean)
        frac_plot.append(frac_comp_mean)
    plt.plot(frac_plot, pq_plot)
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("Pair quality (PQ)")
    plt.grid()
    plt.show()
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("Pair completeness (PC)")
    plt.plot(frac_plot, pc_plot)
    plt.grid()
    plt.show()
    plt.plot(frac_plot, f1_plot)
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("F1 - measure")
    plt.grid()
    plt.show()
    print(f"bands: {test_bands} \n"
          f"fraction: {frac_plot} \n"
          f"pq: {pq_plot} \n"
          f"pc: {pc_plot} \n"
          f"f1: {f1_plot}")
