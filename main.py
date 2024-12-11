from data import collect_json, transform_features, extract_brands, extract_model_words, determine_real_pairs
from lsh import create_signature_matrix, return_candidate_pairs, generate_lsh_summary_and_plots, evaluate_lsh
from msm import perform_msm, evaluate_msm, optimize_epsilon

if __name__ == '__main__':
    AllTv = collect_json()
    all_ids = transform_features(AllTv)
    known_brands = extract_brands(AllTv)
    all_model_words, sets_of_words = extract_model_words(AllTv, known_brands)
    SignatureMatrix = create_signature_matrix(all_model_words=all_model_words,
                                              model_words_per_tv=sets_of_words)
    matches = return_candidate_pairs(bands=15, signature_matrix=SignatureMatrix)
    real_pairs = determine_real_pairs(AllTv)
    #bootstrap 5 times
    for i in range(0,5):
        a =  1

    #evaluate_lsh(real_pairs, matches)
    generate_lsh_summary_and_plots(signature_matrix=SignatureMatrix,
                                   all_tv=AllTv)
    print(matches[0])
    # clusters = perform_msm(matches, AllTv, known_brands, )
    # evaluate_msm(clusters, real_pairs)
    # optimize_epsilon(matches, AllTv, known_brands, real_pairs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
