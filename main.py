from data import collect_json, transform_features, extract_brands, extract_model_words
from lsh import create_signature_matrix, return_potential_matches, generate_lsh_plots
from msm import perform_msm

if __name__ == '__main__':
    AllTv = collect_json()
    all_ids = transform_features(AllTv)
    known_brands = extract_brands(AllTv)
    all_model_words, sets_of_words = extract_model_words(AllTv, known_brands)
    print(sets_of_words[0:5])
    print(len(sets_of_words))
    print(len(all_model_words))
    SignatureMatrix = create_signature_matrix(all_model_words=all_model_words, model_words_per_tv=sets_of_words)
    # print(signature_matrix)
    matches = return_potential_matches(bands=15, signature_matrix=SignatureMatrix)
    print(len(matches))
    matching = 0
    # for j in range(0, len(matches)):
    #     #print(j)
    #     #print(matches[j])
    #     tv_0 = matches[j][0]
    #     for i in range(1, len(matches[j])):
    #         if matches[j][i] - tv_0 == 1:
    #             #print(matches[j])
    #             matching += 1
    #         tv_0 = matches[j][i]
    print(matching)
    # real_pairs = determine_real_pairs(all_tv)
    # evaluate_lsh(real_pairs, matches)
    # generate_lsh_plots(signature_matrix=SignatureMatrix,
    #                    all_tv=AllTv)
    print(matches[0])
    perform_msm(matches, AllTv, known_brands)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
