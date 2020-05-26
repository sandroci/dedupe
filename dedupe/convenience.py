#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from math import radians, cos, sin, asin, sqrt
import collections
import itertools
import sys
from typing import List, Tuple, Dict, Set

from colorama import Fore, Style, init
init(autoreset=True)

import dedupe
from dedupe.core import randomPairs, randomPairsMatch, unique
from dedupe.canonical import getCanonicalRep
from dedupe._typing import Data, TrainingData, RecordDict, TrainingExample, Literal, RecordID


def calculate_haversine_distance(data_pair_1, data_pair_2):
    # convert decimal degrees to radians
    longitude_1 = data_pair_1[0]
    latitude_1 = data_pair_1[1]
    longitude_2 = data_pair_2[0]
    latitude_2 = data_pair_2[1]

    lon1, lat1, lon2, lat2 = map(radians, [longitude_1, latitude_1, longitude_2, latitude_2])

    # haversine formula
    distance_longitude = lon2 - lon1
    distance_lat = lat2 - lat1
    a = sin(distance_lat/2)**2 + cos(lat1) * cos(lat2) * sin(distance_longitude/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers
    distance = c * r
    return distance


def console_label(deduper: dedupe.api.ActiveMatching) -> Tuple:  # pragma: no cover
    '''
    Train a matcher instance (Dedupe, RecordLink, or Gazetteer) from the command line.
    Example

   .. code:: python

      > deduper = dedupe.Dedupe(variables)
      > deduper.prepare_training(data)
      > dedupe.console_label(deduper)
      '''

    finished = False
    use_previous = False
    fields = unique(field.field
                    for field
                    in deduper.data_model.primary_fields)

    buffer_len = 1  # Max number of previous operations
    examples_buffer: List[Tuple[TrainingExample, Literal['match', 'distinct', 'uncertain']]] = []
    uncertain_pairs: List[TrainingExample] = []

    while not finished:
        if use_previous:
            record_pair, _ = examples_buffer.pop(0)
            use_previous = False
        else:
            if not uncertain_pairs:
                uncertain_pairs = deduper.uncertain_pairs()

            try:
                record_pair = uncertain_pairs.pop()
            except IndexError:
                break

        # TODO: return n_match and n_distinct
        n_match = (len(deduper.training_pairs['match']) +
                   sum(label == 'match' for _, label in examples_buffer))
        n_distinct = (len(deduper.training_pairs['distinct']) +
                      sum(label == 'distinct' for _, label in examples_buffer))

        place_of_residence_pair_1  = record_pair[0]['place_of_residence_wgs84_combined']
        place_of_residence_pair_2  = record_pair[1]['place_of_residence_wgs84_combined']
        distance_between_residence_pairs = calculate_haversine_distance(place_of_residence_pair_1, place_of_residence_pair_2)

        place_of_citizenship_pair_1  = record_pair[0]['place_of_citizenship_wgs84_combined']
        place_of_citizenship_pair_2  = record_pair[1]['place_of_citizenship_wgs84_combined']
        if not np.any(np.isnan(place_of_citizenship_pair_1)) and not np.any(np.isnan(place_of_citizenship_pair_2)):
            distance_between_citizenship_pairs = calculate_haversine_distance(place_of_citizenship_pair_1, place_of_citizenship_pair_2)


        print(file=sys.stderr)
        name_to_compare = None
        firstname_to_compare = None
        birthyear_to_compare = None
        sex_to_compare = None
        for index, pair in enumerate(record_pair):
            for field in fields:
                if index == 0:
                    if field == 'place_of_residence_wgs84_combined' or field == 'place_of_citizenship_wgs84_combined':
                        continue
                    elif field == 'name':
                        name_to_compare = pair[field]
                    elif field == 'firstname':
                        firstname_to_compare = pair[field]
                    elif field == 'birthyear':
                        birthyear_to_compare = pair[field]
                    elif field == 'sex':
                        sex_to_compare = pair[field]

                    line = "%s : %s" % (field, pair[field])

                else:
                    if field == 'place_of_residence_wgs84_combined' or field == 'place_of_citizenship_wgs84_combined':
                        continue

                    if field == 'name':
                        if not pair[field]:
                            line = f'{Fore.YELLOW}{field}:{Style.RESET_ALL} {pair[field]}'
                        elif pair[field] != name_to_compare:
                            line = f'{Fore.RED}{field}:{Style.RESET_ALL} {pair[field]}'
                        else:
                            line = f'{Fore.GREEN}{field}:{Style.RESET_ALL} {pair[field]}'

                    elif field == 'firstname':
                        if not pair[field]:
                            line = f'{Fore.YELLOW}{field}:{Style.RESET_ALL} {pair[field]}'
                        elif pair[field] != firstname_to_compare:
                            line = f'{Fore.RED}{field}:{Style.RESET_ALL} {pair[field]}'
                        else:
                            line = f'{Fore.GREEN}{field}:{Style.RESET_ALL} {pair[field]}'

                    elif field == 'birthyear':
                        if not pair[field]:
                            line = f'{Fore.YELLOW}{field}:{Style.RESET_ALL} {pair[field]}'
                        elif pair[field] != birthyear_to_compare:
                            line = f'{Fore.RED}{field}:{Style.RESET_ALL} {pair[field]}'
                        else:
                            line = f'{Fore.GREEN}{field}:{Style.RESET_ALL} {pair[field]}'

                    elif field == 'sex':
                        if not pair[field]:
                            line = f'{Fore.YELLOW}{field}:{Style.RESET_ALL} {pair[field]}'
                        elif pair[field] != sex_to_compare:
                            line = f'{Fore.RED}{field}:{Style.RESET_ALL} {pair[field]}'
                        else:
                            line = f'{Fore.GREEN}{field}:{Style.RESET_ALL} {pair[field]}'


                print(line, file=sys.stderr)

            print(file=sys.stderr)

        distance_residence_output = f'distance residence places: {int(distance_between_residence_pairs)} km'
        print(distance_residence_output, file=sys.stderr)
        distance_citizenship_output = f'distance citizenship places: {int(distance_between_citizenship_pairs)} km'
        print(distance_citizenship_output, file=sys.stderr)

        print(file=sys.stderr)
        print('Do these records refer to the same person?', file=sys.stderr)
        print(file=sys.stderr)

        valid_response = False
        user_input = ''
        while not valid_response:
            if examples_buffer:
                prompt = '(y)es / (n)o / (u)nsure / (f)inished / (p)revious'
                valid_responses = {'y', 'n', 'u', 'f', 'p'}
            else:
                prompt = '(y)es / (n)o / (u)nsure / (f)inished'
                valid_responses = {'y', 'n', 'u', 'f'}

            print(prompt, file=sys.stderr)
            user_input = input()
            if user_input in valid_responses:
                valid_response = True

        if user_input == 'y':
            examples_buffer.insert(0, (record_pair, 'match'))
        elif user_input == 'n':
            examples_buffer.insert(0, (record_pair, 'distinct'))
        elif user_input == 'u':
            examples_buffer.insert(0, (record_pair, 'uncertain'))
        elif user_input == 'f':
            print('Finished labeling', file=sys.stderr)
            finished = True
        elif user_input == 'p':
            use_previous = True
            uncertain_pairs.append(record_pair)

        if len(examples_buffer) > buffer_len:
            record_pair, label = examples_buffer.pop()
            if label in {'distinct', 'match'}:

                examples: TrainingData
                examples = {'distinct': [],
                            'match': []}
                examples[label].append(record_pair)  # type: ignore
                deduper.mark_pairs(examples)

    for record_pair, label in examples_buffer:
        if label in ['distinct', 'match']:

            exmples: TrainingData
            examples = {'distinct': [], 'match': []}
            examples[label].append(record_pair)  # type: ignore
            deduper.mark_pairs(examples)

    return (n_match, n_distinct)


def training_data_link(data_1: Data,
                       data_2: Data,
                       common_key: str,
                       training_size: int = 50000) -> TrainingData:  # pragma: nocover
    '''
    Construct training data for consumption by the func:`mark_pairs`
    method from already linked datasets.

    Args:

        data_1: Dictionary of records from first dataset, where the
        keys are record_ids and the values are dictionaries
        with the keys being field names
        data_2: Dictionary of records from second dataset, same form as
        data_1
        common_key: The name of the record field that uniquely identifies
        a match
        training_size: the rough limit of the number of training examples,
        defaults to 50000

    .. note::

         Every match must be identified by the sharing of a common key.
         This function assumes that if two records do not share a common key
         then they are distinct records.
         '''

    identified_records: Dict[str, Tuple[List[RecordID], List[RecordID]]]
    identified_records = collections.defaultdict(lambda: ([], []))
    matched_pairs: Set[Tuple[RecordID, RecordID]] = set()
    distinct_pairs: Set[Tuple[RecordID, RecordID]] = set()

    for record_id, record in data_1.items():
        identified_records[record[common_key]][0].append(record_id)

    for record_id, record in data_2.items():
        identified_records[record[common_key]][1].append(record_id)

    for keys_1, keys_2 in identified_records.values():
        if keys_1 and keys_2:
            matched_pairs.update(itertools.product(keys_1, keys_2))

    keys_1 = list(data_1.keys())
    keys_2 = list(data_2.keys())

    random_pairs = [(keys_1[i], keys_2[j])
                    for i, j
                    in randomPairsMatch(len(data_1), len(data_2),
                                        training_size)]

    distinct_pairs = {
        pair for pair in random_pairs if pair not in matched_pairs}

    matched_records = [(data_1[key_1], data_2[key_2])
                       for key_1, key_2 in matched_pairs]
    distinct_records = [(data_1[key_1], data_2[key_2])
                        for key_1, key_2 in distinct_pairs]

    training_pairs: TrainingData
    training_pairs = {'match': matched_records,
                      'distinct': distinct_records}

    return training_pairs


def training_data_dedupe(data: Data,
                         common_key: str,
                         training_size: int = 50000) -> TrainingData:  # pragma: nocover
    '''
    Construct training data for consumption by the func:`mark_pairs`
    method from an already deduplicated dataset.

    Args:

        data: Dictionary of records where the keys are record_ids and
        the values are dictionaries with the keys being field names
        common_key: The name of the record field that uniquely identifies
        a match
        training_size: the rough limit of the number of training examples,
        defaults to 50000

    .. note::

         Every match must be identified by the sharing of a common key.
         This function assumes that if two records do not share a common key
         then they are distinct records.
         '''

    identified_records: Dict[str, List[RecordID]]
    identified_records = collections.defaultdict(list)
    matched_pairs: Set[Tuple[RecordID, RecordID]] = set()
    distinct_pairs: Set[Tuple[RecordID, RecordID]] = set()
    unique_record_ids: Set[RecordID] = set()

    # a list of record_ids associated with each common_key
    for record_id, record in data.items():
        unique_record_ids.add(record_id)
        identified_records[record[common_key]].append(record_id)

    # all combinations of matched_pairs from each common_key group
    for record_ids in identified_records.values():
        if len(record_ids) > 1:
            matched_pairs.update(itertools.combinations(sorted(record_ids), 2))  # type: ignore

    # calculate indices using dedupe.core.randomPairs to avoid
    # the memory cost of enumerating all possible pairs
    unique_record_ids_l = list(unique_record_ids)
    pair_indices = randomPairs(len(unique_record_ids), training_size)
    distinct_pairs = set()
    for i, j in pair_indices:
        distinct_pairs.add((unique_record_ids_l[i],
                            unique_record_ids_l[j]))

    distinct_pairs -= matched_pairs

    matched_records = [(data[key_1], data[key_2])
                       for key_1, key_2 in matched_pairs]

    distinct_records = [(data[key_1], data[key_2])
                        for key_1, key_2 in distinct_pairs]

    training_pairs: TrainingData
    training_pairs = {'match': matched_records,
                      'distinct': distinct_records}

    return training_pairs


def canonicalize(record_cluster: List[RecordDict]) -> RecordDict:  # pragma: nocover
    """
    Constructs a canonical representation of a duplicate cluster by
    finding canonical values for each field

    Args:
        record_cluster: A list of records within a duplicate cluster, where
        the records are dictionaries with field
        names as keys and field values as values

    """
    return getCanonicalRep(record_cluster)
