import json
import re


def collect_json() -> list[dict]:
    """
    Collect all data from json and seperate tv's witin one modelID

    :return: list of dict
    """
    with open(
            '/Users/koenboerjan/Documents/Econometrie/Master/Computer science for bussines analytics/Paper/TVs-all-merged.json',
            'r') as file:
        json_data = json.load(file)
    data = []
    for model_id in json_data.keys():
        # if len(json_data[model_id]) > 2:
        #     print(json_data[model_id])
        data.extend(json_data[model_id])
    return data


def transform_features(data: list[dict]) -> list[dict]:
    return [line['modelID'] for line in data]


def extract_brands(data: list[dict]) -> set[str]:
    """
    Collect different brand names from data based on feature with key 'Brand'

    :param data: all data from json file
    :return: set of all brands
    """
    different_brands = []
    for shop_object in data:
        try:
            brand_feature = shop_object['featuresMap']['Brand']
        except:
            brand_feature = ''
        if brand_feature and not (' ' in brand_feature):
            different_brands.append(brand_feature.lower())
    return set(different_brands)


def extract_model_words(data: list[dict], known_brands: set, include_feature: bool = False) -> (set[str], list[set[str]]):
    """
    Collect all model words from list of dict with tv_data, model words are defined by specific regex, containing both
    numerical as alphabetical values. Possibility to include data from features as well.

    :param data: all tv_data in list of dict formate
    :param known_brands: set of brand names
    :param include_feature: boolean if including model words from features.
    :return: set of all model words and list of set of model words per tv
    """
    model_words_regex = re.compile('[a-zA-Z0-9]*(([0-9]+[ˆ0-9,]+)|([ˆ0-9,]+[0-9]+))[a-zA-Z0-9]*')
    mw_per_product = []
    all_model_words = set()
    for item in data:
        model_set = set(item['title'].lower().split())
        if include_feature:
            for feature_key in item['featuresMap'].keys():
                features_value = item['featuresMap'][feature_key].lower().split()
                model_set.update(set(features_value))

        brand = model_set & known_brands
        model_words = set(filter(model_words_regex.match, model_set))
        model_words = reformat_model_words(model_words)
        model_words.update(brand)
        mw_per_product.append(model_words)
        all_model_words = all_model_words | model_words
    return all_model_words, mw_per_product


def reformat_model_words(unformatted: set[str]) -> set[str]:
    """
    Reformat model words by removing special characters and units.

    :param unformatted: set of unedited model words
    :return: set of formated model words
    """
    find_special_char_words = re.compile('[^a-zA-Z0-9]+[a-zA-Z0-9]*|[a-zA-Z0-9]*[^a-zA-Z0-9]+')
    special_char_words = set(filter(find_special_char_words.match, unformatted))
    without_special_char = unformatted.difference(special_char_words)

    find_special_char = re.compile('[^a-zA-Z0-9]+')
    for word in special_char_words:
        for character in set(filter(find_special_char.match, word)):
            word = word.replace(character, '')
        without_special_char.add(word)

    # remove units
    find_unit_words = re.compile('^[0-9]+[a-zA-Z]+$')
    unit_words = (set(filter(find_unit_words.match, without_special_char)))
    formatted = without_special_char.difference(unit_words)

    find_char = re.compile('[a-zA-Z]')
    for word in unit_words:
        for letter in set(filter(find_char.match, word)):
            word = word.replace(letter, '')
        formatted.add(word)

    return formatted


def determine_real_pairs(complete_dataset: list[dict]) -> list[set[int]]:
    """
    Find all real pairs within the dataset based on same model id. Dataset is assumed to be ordered,
    two items with same model id are next to each other.

    :param complete_dataset: ordered data of tvs
    :return: list of all different sets of pairs
    """
    matching_models = []
    last_model_id = complete_dataset[0]['modelID']
    index = 1
    while index < len(complete_dataset):
        matching_index = [index - 1]
        while last_model_id == complete_dataset[index]['modelID']:
            matching_index.append(index)
            last_model_id = complete_dataset[index]['modelID']
            index += 1
        if len(matching_index) > 1:
            matching_models.append(matching_index)
        last_model_id = complete_dataset[index]['modelID']
        index += 1

    one_on_one_real_pairs = []
    for pairs in matching_models:
        if len(pairs) > 2:
            for i1 in range(0, len(pairs)):
                for i2 in range(i1 + 1, len(pairs)):
                    one_on_one_real_pairs.append({pairs[i1], pairs[i2]})
        else:
            one_on_one_real_pairs.append(set(pairs))
    return one_on_one_real_pairs
