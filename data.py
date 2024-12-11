import json
import re


def collect_json() -> list[dict]:
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


def determine_real_pairs(complete_dataset: list[dict]):
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
    return matching_models
