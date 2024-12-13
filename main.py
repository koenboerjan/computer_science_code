from data import collect_json, transform_features, extract_brands, extract_model_words, determine_real_duplicates, \
    bootstrap_sample
from lsh import create_signature_matrix, return_candidate_pairs, generate_lsh_summary_and_plots, evaluate_lsh
from msm import perform_msm, evaluate_msm, optimize_epsilon
import random
from numpy.ma.extras import average
import matplotlib.pyplot as plt


def create_result_msm(all_data: list[dict], known_brands: set[str]):
    """
    Print result of msm without lsh, based on fully random bootstrap sample.

    :param all_data: complete dataset
    :param known_brands: set of all known brands
    :return:
    """
    tv_ids = [index for index in range(0, len(all_data))]
    random.seed(1)
    bootstrap_results = []
    for i in range(0, 5):
        print(i)
        train_sample = sorted(random.sample(tv_ids, round(0.63 * len(all_data))))
        test_sample = [tv_id for tv_id in tv_ids if tv_id not in train_sample]
        # train
        train_tvs = [all_data[i] for i in range(0, len(all_data)) if i in train_sample]
        real_pairs = determine_real_duplicates(train_tvs)
        epsilon = optimize_epsilon([[index for index in range(0, len(train_tvs))]], train_tvs, known_brands, real_pairs)
        # test
        print(epsilon)
        test_tvs = [all_data[i] for i in range(0, len(AllTv)) if i in test_sample]
        real_pairs = determine_real_duplicates(test_tvs)
        clustered = perform_msm([[index for index in range(0, len(test_tvs))]], test_tvs, known_brands, epsilon)
        bootstrap_results.append(evaluate_msm(clustered, real_pairs, len(test_tvs)))
    print(bootstrap_results)
    return []

def create_result_msm_bootstrap(known_brands: set[str]):
    """
    Print result of msm without lsh, based on model id bootstrap samples.

    :param known_brands: set of all known brands
    :return:
    """
    random.seed(1)
    bootstrap_results = []
    for i in range(0, 5):
        print(i)
        train_sample, test_sample = bootstrap_sample()

        # train
        real_pairs = determine_real_duplicates(train_sample)
        epsilon = optimize_epsilon([[index for index in range(0, len(train_sample))]], train_sample, known_brands, real_pairs)
        # test
        print(epsilon)
        real_pairs = determine_real_duplicates(test_sample)
        clustered = perform_msm([[index for index in range(0, len(test_sample))]], test_sample, known_brands, epsilon)
        bootstrap_results.append(evaluate_msm(clustered, real_pairs, len(test_sample)))
        print(bootstrap_results)
    print(bootstrap_results)
    return []


def create_result_msm_lsh_fully_random(all_data: list[dict], known_brands: set[str]):
    """
    Create summary of msm lsh combined with data sampled fully random.

    :param all_data: full dataset
    :param known_brands: set of all known brands
    :return:
    """
    random.seed(1)
    tv_ids = [index for index in range(0, len(AllTv))]
    mean_f1_for_plot = []
    mean_frac_for_plot = []
    for n in [4, 5, 6, 10, 12, 15, 20, 30]:
        print(n)
        lsh_results = []
        bootstrap_results = []
        for i in range(0, 5):
            print(i)
            train_sample = sorted(random.sample(tv_ids, round(0.63 * len(AllTv))))
            test_sample = [tv_id for tv_id in tv_ids if tv_id not in train_sample]

            # train
            train_tvs = [all_data[i] for i in range(0, len(all_data)) if i in train_sample]
            all_model_words, sets_of_words = extract_model_words(data=train_tvs,
                                                                 known_brands=known_brands)
            signature_matrix = create_signature_matrix(all_model_words=all_model_words,
                                                       model_words_per_tv=sets_of_words)
            matches = return_candidate_pairs(bands=n,
                                             signature_matrix=signature_matrix)
            real_pairs_train = determine_real_duplicates(train_tvs)
            epsilon = optimize_epsilon(matches, train_tvs, known_brands, real_pairs_train)

            # test
            print(epsilon)
            test_tvs = [all_data[i] for i in range(0, len(all_data)) if i in test_sample]
            all_model_words, sets_of_words = extract_model_words(data=test_tvs,
                                                                 known_brands=known_brands)
            signature_matrix = create_signature_matrix(all_model_words=all_model_words,
                                                       model_words_per_tv=sets_of_words)
            matches = return_candidate_pairs(bands=n,
                                             signature_matrix=signature_matrix)
            real_pairs_test = determine_real_duplicates(test_tvs)
            lsh_results.append(evaluate_lsh(real_pairs_test, matches, len(test_tvs)))
            clustered = perform_msm(matches, test_tvs, known_brands, epsilon)
            bootstrap_results.append(evaluate_msm(clustered, real_pairs_test, len(test_tvs)))
            print(bootstrap_results)
        mean_f1_for_plot.append(average([value[2] for value in bootstrap_results]))
        mean_frac_for_plot.append(average([value[3] for value in lsh_results]))
        print(mean_f1_for_plot)
        print(mean_frac_for_plot)
    plt.plot(mean_frac_for_plot, mean_f1_for_plot)
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("F1 - measure")
    plt.grid()
    plt.show()
    return []


def create_result_msm_lsh_bootstrap_model_id(known_brands: set[str]):
    """
    Creates plot of F1 measure for msm and lsh combined where bootstrap is sampled based ond model id
    Also prints the values of the f1 measure together with fraction of comparisons.

    :param known_brands: set of all known brands
    :return:
    """
    random.seed(1)
    mean_f1_for_plot = []
    mean_frac_for_plot = []
    for n in [5, 6, 10, 12, 15, 20, 30]:
        print(n)
        lsh_results = []
        bootstrap_results = []
        for i in range(0, 5):
            print(i)
            training_sample, test_sample = bootstrap_sample()

            # train
            all_model_words, sets_of_words = extract_model_words(data=training_sample,
                                                                 known_brands=known_brands)
            signature_matrix = create_signature_matrix(all_model_words=all_model_words,
                                                       model_words_per_tv=sets_of_words)
            matches = return_candidate_pairs(bands=n,
                                             signature_matrix=signature_matrix)
            real_pairs_train = determine_real_duplicates(training_sample)
            epsilon = optimize_epsilon(matches, training_sample, known_brands, real_pairs_train)

            # test
            print(epsilon)
            all_model_words, sets_of_words = extract_model_words(data=test_sample,
                                                                 known_brands=known_brands)
            signature_matrix = create_signature_matrix(all_model_words=all_model_words,
                                                       model_words_per_tv=sets_of_words)
            matches = return_candidate_pairs(bands=n,
                                             signature_matrix=signature_matrix)
            real_pairs_test = determine_real_duplicates(test_sample)

            lsh_results.append(evaluate_lsh(real_pairs_test, matches, len(test_sample)))
            clustered = perform_msm(matches, test_sample, known_brands, epsilon)
            bootstrap_results.append(evaluate_msm(clustered, real_pairs_test, len(test_sample)))
        mean_f1_for_plot.append(average([value[2] for value in bootstrap_results]))
        mean_frac_for_plot.append(average([value[3] for value in lsh_results]))
        print(mean_f1_for_plot)
        print(mean_frac_for_plot)
    plt.plot(mean_frac_for_plot, mean_f1_for_plot)
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("F1 - measure")
    plt.grid()
    plt.show()
    return 0



def create_result_lsh(all_data: list[dict], known_brands: set[str]):
    """
    Prints plot of LSH result for different sizes of bands

    :param all_data: dataset
    :param known_brands: set of all known brands
    :return: nothing
    """
    all_model_words, sets_of_words = extract_model_words(data=all_data,
                                                         known_brands=known_brands)
    signature_matrix = create_signature_matrix(all_model_words=all_model_words,
                                               model_words_per_tv=sets_of_words)
    generate_lsh_summary_and_plots(signature_matrix=signature_matrix,
                                   all_tv=all_data)


def compare_bootstrapping_methods():
    """
    Compares the two bootstrapping methods and prints a summary

    :return: nothing
    """
    random.seed(1)
    full_pairs = 0
    average_fully_random = 0
    average_model_id_random = 0
    for i in range(0,5):
        data_sample = collect_json()
        real_pairs_full = determine_real_duplicates(data_sample)
        print(f"Pairs in full dataset: {len(real_pairs_full)}")
        full_pairs = len(real_pairs_full)

        #Fully random
        tv_ids = [index for index in range(0, len(AllTv))]
        test_sample = sorted(random.sample(tv_ids, round(0.37 * len(data_sample))))
        test_tvs = [data_sample[i] for i in range(0, len(data_sample)) if i in test_sample]
        real_pairs = determine_real_duplicates(test_tvs)
        print(f"Pairs in fully random test set: {len(real_pairs)}")
        average_fully_random += 1/5 * len(real_pairs)

        #Random model_ids
        train_sample, test_sample = bootstrap_sample()
        real_pairs = determine_real_duplicates(test_sample)
        print(f"Pairs in random model id's test set: {len(real_pairs)}")
        average_model_id_random += 1/5 * len(real_pairs)
    print(f"Pairs in full dataset: {full_pairs}")
    print(f"Average pairs in fully random test set: {average_fully_random}")
    print(f"Average airs in random model id's test set: {average_model_id_random}")

if __name__ == '__main__':
    random.seed(1)
    AllTv = collect_json()
    all_ids = transform_features(AllTv)
    KnownBrands = extract_brands(AllTv)


    create_result_msm_bootstrap(known_brands=KnownBrands)
    create_result_msm(all_data=AllTv,
                      known_brands=KnownBrands)
    compare_bootstrapping_methods()
    create_result_msm_lsh_fully_random(all_data=AllTv,
                                       known_brands=KnownBrands)
    create_result_msm_lsh_bootstrap_model_id(known_brands=KnownBrands)
    create_result_lsh(all_data=AllTv,
                      known_brands=KnownBrands)

