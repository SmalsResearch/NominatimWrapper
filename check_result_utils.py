# -*- coding: utf-8 -*-
"""
Functions used to check OSM results

@author: Vandy Berten (vandy.berten@smals.be)
"""

# pylint: disable=invalid-name
# pylint: disable=line-too-long


import unicodedata
import os

import pandas as pd
#import numpy as np

import jellyfish

from config import (addr_key_field,
                    street_field,
                    housenbr_field,
                    postcode_field,
                    city_field,
                    country_field,
                    similarity_threshold)

from base import vlog, log, get_osm_details


def remove_accents(input_str):
    """
    Remove accents and other diactricis symbols

    Parameters
    ----------
    input_str : str
        string with accents to remove.

    Returns
    -------
    str
        input_str without any accents and other diactritics symbols.

    """
    if pd.isnull(input_str):
        return None

    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


# In[ ]:


def house_number_compare(n1, n2):
    """

    Compare two (identical size) pd.Series of house numbers
    - if n1 == n2 (and not empty) --> 1
    - else we split n1 and n2 in chunks of numbers :
        - if first chunk of n1 = second chunk of n2 -->  0.8  (e.g : 10 vs 10-12)
        - else second chunk of n1 = first chunk of n2 --> 0.8 (e.g. 10-12 vs 12)
    - else if only numbers are equal (and not empty) --> 0.5  (e.g., '10a' vs '10 B' vs '10')
    Parameters
    ----------
    n1 : pd.Series
        series of housenumber
    n2 : pd.Series
        series of housenumber

    Returns
    -------
    res : pd.Series
        Pandas series, same size as n1 or n2, with values 0, 0.5, 0.8 or 1.

    """

    assert n1.shape[0] == n2.shape[0]

    n1 = n1.fillna("").astype(str).str.strip()
    n2 = n2.fillna("").astype(str).str.strip()

    res= ((n1==n2) & (n1.str.len()>0)).astype(float)

    n1_split = n1.str.split("[^0-9]", expand=True)
    n2_split = n2.str.split("[^0-9]", expand=True)

    if n2_split.shape[1]>1:
        res[(res == 0) & ((n1_split[0] == n2_split[1]) & (n2_split[1].str.len()>0))] = 0.8

    if n1_split.shape[1]>1:
        res[(res == 0) & ((n1_split[1] == n2_split[0]) & (n1_split[1].str.len()>0))] = 0.8

    res[(res == 0) & (n1.str.replace("[^0-9]", "",regex=True) == n2.str.replace("[^0-9]", "",regex=True)) & ( n1.str.len()>0) & ( n2.str.len()>0)] = 0.5

    return res


# In[ ]:


def postcode_compare(s1, s2):
    """
    Compare postcodes within two (identical size) pandas series.

    If the postcodes in both records are identical, the similarity
    is 1. If the first two values agree and the last two don't, then
    the similarity is 0.5. Otherwise, the similarity is 0.

    Parameters
    ----------
    s1 : pd.Series
        series of postcodes.
    s2 : pd.Series
        series of postcodes.

    Returns
    -------
    sim : pd.Series
        Pandas series, same size as s1 or s2, with values 0, 0.5, or 1.

    """

    assert s1.shape[0] == s2.shape[0]

    s1 = s1.fillna("").astype(str).str.replace("^[A-Z]-?", "",regex=True)
    s2 = s2.fillna("").astype(str).str.replace("^[A-Z]-?", "",regex=True)

    # check if the postcode are identical (return 1 or 0)
    sim = (s1 == s2).astype(float)
    # one is missing
    sim[(sim == 0) & ((s1.fillna("").str.len()==0) | (s2.fillna("").str.len()==0))] = 0.1

    # check the first 2 numbers of the distinct comparisons

    sim[(sim == 0) & (s1.str[0:2] == s2.str[0:2])] = 0.5
    sim[(sim == 0) & (s1.str[0:1] == s2.str[0:1])] = 0.3


    return sim


# In[ ]:


def levenshtein_similarity(str1, str2):
    """
    Compute Damerau-Levenshtein similarity between two strings str1 and str2,
    i.e., 1 - dist(str1, str2)/N,
    where
    - dist(str1, str2) is the Damerau-Levenshtein distance,
    - N is the max of len(str1) and len(str2)

    Parameters
    ----------
    str1 : str
        string to compare 1.
    str2 : str
        string to compare 1.

    Returns
    -------
    float
        a value between 0 (nothing in common) and 1 (exact similarity).

    """
    return 1 - jellyfish.damerau_levenshtein_distance(str1, str2)/max(len(str1), len(str2)) if (len(str1) > 0 or len(str2) > 0  ) else 0.0


# In[ ]:





# In[ ]:


def inclusion_test(s1, s2):
    """
    Check that a string s1 is equal to another string s2, except that s2 contains an additional substring
    Example : "Avenue C Berten" vs "Avenue Clovis Berten"

    Parameters
    ----------
    s1 : str
        string to compare 1.
    s2 : TYPE
        string to compare 2.

    Returns
    -------
    0 (no inclustion) or 1 (inclusion).

    """


    l_pref = len(os.path.commonprefix([s1, s2]))
    l_suf =  len(os.path.commonprefix([s1[::-1], s2[::-1]]))

    res = 1 if (l_pref>0)  and (l_suf > 0) and (l_pref+l_suf >= min(len(s1), len(s2))) else 0
#     if res == 1:
#         print(s1, s2, res)
    return res



# In[ ]:


def fingerprint(column):
    """
    Compute the fingerprint of each string in a pandas Series, assuming it was
    cleansed beforehand (contains only uppercases and no accents):
        - keep only alphabetic characters
        - order words
        - remove duplicates

    Example: "RUE DU BAS DU VILLAGE" --> "BAS DU RUE VILLAGE"

    Parameters
    ----------
    column : pd.Series
        Series of (cleansed) strings.

    Returns
    -------
    pd.Series.

    """
    cleaned_column = column.fillna("")
#     cleaned_column = cleaned_column.str.upper().apply(remove_accents)
    cleaned_column = cleaned_column.str.replace("[^A-Z]", " ",regex=True)
    cleaned_column = cleaned_column.str.strip()
    cleaned_column = cleaned_column.str.split("[ ]+")

    cleaned_column = cleaned_column.apply(lambda x: sorted(list(set(x))))
    cleaned_column = cleaned_column.apply(" ".join)
    return cleaned_column


# In[ ]:



# TODO : replacement seulement si dans les 2 inputs en même temps
# --> Pour éviter que "Avenue Louise" et "Place Louise" aient une similarité de 100%

def_street_compare_removes = [r"\([A-Z.]+\)",
                          # r"[^A-Z ]+",
                          r"\b(AVENUE|RUE|CHAUSSEE|BOULEVARD|PLACE)\b",
                          r"(STRAAT|LAAN|STEENWEG|WEG)\b"
                         ]

dontwatchthis = "DONOTCONSIDERTHISSTRING"

def _street_compare(street1, street2, compare_algo, street_compare_removes):
    """
    Technical function used by "street_compare". Will compare two columns
    containing street names.
    It first remove all regex from street_compare_removes
    Then gives a score to each pair of streets (str1, str2):
        - 0 if length diff is > 10
        - 0 if one is equal to "dontwatchthis" (see street_compare)
        - 1 if both are empty
        - compare_algo(str1, str2) otherwise

    Parameters
    ----------
    street1 : pd.Series
        First column.
    street2 : pd.Series
        Second column.
    compare_algo : function(str, str)
        distance function applied on 2 strings.
    street_compare_removes : list
        list of regexps to remove in both columns.

    Returns
    -------
    res : pd.Series
        values between 0 and 1 (both inclusive)

    """

    assert street1.shape[0] == street2.shape[0]

    streets = pd.DataFrame()
    streets["STR1"] = street1
    streets["STR2"] = street2

    for i in ["STR1", "STR2"]:
        for scr in street_compare_removes:
            streets[i] = streets[i].str.replace(scr, "",regex=True)

        streets[i] = streets[i].str.strip().str.replace(" [ ]+", " ",regex=True)

    # if diff length > 10 : --> 0, othewise : 2
    res = (streets.STR1.fillna("").str.len() - streets.STR1.fillna("").str.len() < 10).astype(float) *2

    # If one is equal to "dontwatchthis" : 0
    res[(res == 2) & ((streets.STR1 == dontwatchthis) | (streets.STR2 == dontwatchthis)) ] = 0.0

    # if both is empty : 1
    res[(res == 2) & (streets.STR1.fillna("") == "") & (streets.STR2.fillna("") == "") ] = 1.0

    # Otherwise (still == 2): compute distance


    if (res == 2).any():
        res[res == 2] = streets[res == 2].fillna("").apply(lambda row : compare_algo(row.STR1, row.STR2), axis=1)

    return res


def street_compare(street1, street2):
    """
    Compare streets in two pandas (identical size) columns
    For Brussels (or bilingual regions), we typically get "Avenue Louise - Louizalaan" for street
    We also often get (in input) streets like "Bruxelles, Avenue Louise", or "Avenue Louise, 10"

    We split then street1 (resp. street2) in street1_split (resp.street2_split)
    into two columns, one with the text before "," or " - ", the other one with
    the text after this token (or "dontwatchthis" if the cell does not contain
    the token).

    We then clean (remove accents, set upper case, remove anything except letters
    and spaces) all 4 columns and, for the 4 combinations street1_split[0 or 1]
    to street2_split[0 or 1], apply :
        - levenshtein_similarity
        - inclusion_test
        - compage_algo on fingerprint

    For each row, we keep the best (=max) of those 12 values

    Parameters
    ----------
    street1 : pd.Series
        a column with street names.
    street2 : pd.Series
        a column with street names.

    Returns
    -------
    pd.Series
        A columns with values between 0 and 1 (both inclusive), where the ith
        value give the similarity between street1.iloc[ith] and street2.iloc[ith]
    """

    if street1.shape[0] == 0:
        return pd.Series(index=street1.index, dtype=str)

    assert street1.shape[0] == street2.shape[0]


    # For Brussels (or bilingual regions), we typically get "Avenue Louise - Louizalaan" for street
    # We also often get (in input) streets like "Bruxelles, Avenue Louise", or "Avenue Louise, 10"
    # Set "dontwatchthis" for values that do not split, but where a column appear because of other values being split

    street1_split = street1.fillna("").str.replace(",", " - ",regex=True).str.split(" - ", expand=True).fillna(dontwatchthis)
    street2_split = street2.fillna("").str.replace(",", " - ",regex=True).str.split(" - ", expand=True).fillna(dontwatchthis)

    #display(pd.concat([street1, street2], axis=1))
    #display(pd.concat([street_split_a, street_split_b], axis=1))


    fingerprints1 = pd.DataFrame(columns = street1_split.columns)
    for ai in range(street1_split.shape[1]):
        street1_split[ai] = street1_split[ai].str.upper().apply(remove_accents)
        street1_split[ai] = street1_split[ai].str.replace( r"[^A-Z ]+", " ",regex=True).str.replace(" [ ]+", " ",regex=True).str.strip()

        fingerprints1[ai]= fingerprint(street1_split[ai])

    fingerprints2 = pd.DataFrame(columns = street2_split.columns)
    for bi in range(street2_split.shape[1]):
        street2_split[bi] = street2_split[bi].str.upper().apply(remove_accents)
        street2_split[bi] = street2_split[bi].str.replace( r"[^A-Z ]+", " ",regex=True).str.replace(" [ ]+", " ",regex=True).str.strip()
        fingerprints2[bi]= fingerprint(street2_split[bi])



    street_distances = pd.DataFrame()
    for ai in range(street1_split.shape[1]):
        str_a = street1_split[ai]#.str.upper().apply(remove_accents)
        # str_a = str_a.str.replace( r"[^A-Z ]+", " ").str.replace(" [ ]+", " ").str.strip()

        for bi in range(street2_split.shape[1]):
            str_b = street2_split[bi]#.str.upper().apply(remove_accents)
            # str_b = str_b.str.replace( r"[^A-Z ]+", " ").str.replace(" [ ]+", " ").str.strip()

            street_distances[f"sim_street_a{ai}b{bi}"] = _street_compare(str_a,
                                                                         str_b,
                                                                         compare_algo=levenshtein_similarity,
                                                                         street_compare_removes=def_street_compare_removes)
            # to calculate (strict) inclusion, we do not remove "street words" (Rue, Avenue ...)
            street_distances[f"inc_street_a{ai}b{bi}"] = _street_compare(str_a,
                                                                         str_b,
                                                                         compare_algo=inclusion_test,
                                                                         street_compare_removes=[])

            #fgpta = fingerprint(str_a)
            #fgptb = fingerprint(str_b)

            street_distances[f"fing_street_a{ai}b{bi}"] =  _street_compare(fingerprints1[ai],
                                                                           fingerprints2[bi],
                                                                           compare_algo=levenshtein_similarity,
                                                                           street_compare_removes=def_street_compare_removes)

    street_distances["sim_street"] =  street_distances[filter(lambda x: "sim_street_a" in x
                                                              or "inc_street_a" in x
                                                              or "fing_street_a" in x, street_distances)].max(axis=1)
#     display(street_distances[street1.fillna("").str.contains("AUTOSNELWEGEN") ])
    vlog(f"Street compare: {street1.name}, {street2.name}")
    vlog(pd.concat([street1_split, street2_split, street_distances], axis=1))
    return street_distances["sim_street"]


# In[ ]:





# In[ ]:


def city_compare(city1, city2, compare_algo = levenshtein_similarity):
    """
    compage cities from two pandas (identical size) columns

    Parameters
    ----------
    city1 : pd.Series
        a column with city names.
    city2 : pd.Series
        a column with city names.
    compare_algo : TYPE, optional
        DESCRIPTION. The default is levenshtein_similarity.

    Returns
    -------
    pd.Series
        A columns with values between 0 and 1 (both inclusive), where the ith
        value give the similarity between city1.iloc[ith] and city2.iloc[ith]

    """

    assert city1.shape[0] == city1.shape[0]

    cities = pd.DataFrame()
    cities["CITY1"] = city1
    cities["CITY2"] = city2

    for i in ["CITY1", "CITY2"]:
        cities[i] = cities[i].str.upper().apply(remove_accents)
        cities[i] = cities[i].str.strip().str.replace(" [ ]+", " ",regex=True)

    return cities.fillna("").apply(lambda row : compare_algo(row.CITY1, row.CITY2), axis=1)



def ignore_mismatch_keep_bests(addr_matches,
                               street_fields_a, housenbr_field_a,
                               postcode_field_a, city_field_a,
                               street_field_b, housenbr_field_b,
                               postcode_field_b, city_field_b,
                               max_res=1, secondary_sort_field = ("metadata", "osm_order")):
    """
    Compare input address with output result.
    We put in "keep":
        - The first valid result for each input address (if any)
    We put in "reject":
        - All non valid results (with reject_reason='mismatch')
        - All valid results but the first one (with reject_reason='tail')


    Parameters
    ----------
    addr_matches : pd.DataFrame
    street_fields_a : str
    housenbr_field_a : str
    postcode_field_a : str
    city_field_a : str
    street_field_b : str
    housenbr_field_b : str
    postcode_field_b : str
    city_field_b : str
    max_res : int, optional
    secondary_sort_field : str, optional

    Returns
    -------
    keep: pd.DataFrame
        DESCRIPTION.
    reject: pd.DataFrame
        DESCRIPTION.

    """

    if addr_matches.shape[0] == 0:
        return addr_matches, addr_matches

    distances = pd.DataFrame(index=addr_matches.index)

    street_b = addr_matches[street_field_b]

    distances[("check", "sim_street")] = -1
    distances[("check","sim_street_which")]=""
    distances.columns = pd.MultiIndex.from_tuples(distances.columns, names=["L0", "L1"])

    vlog("Will compare streets")
    for street_field_a in street_fields_a :
        # Only compute a new street distance if the computed distance is below the threshold so far
        x = distances[("check","sim_street")] < similarity_threshold

        distances[("check","sim_street")] = distances[("check","sim_street")].where(~x, street_compare(addr_matches[street_field_a][x].fillna(""), street_b[x]))

        distances[("check","sim_street_which")] =  distances[("check","sim_street_which")].where(~x,street_field_a[1]) # last field that has been compared

    wsu = " ; ".join([f"{r}: {c}" for r, c in distances[distances[("check", "sim_street")] >= similarity_threshold][("check","sim_street_which")].value_counts().items()])
    vlog(f"Which street used: {wsu}")


    distances[("check","sim_house_number")] = house_number_compare(addr_matches[housenbr_field_a].fillna(""), addr_matches[housenbr_field_b].fillna(""))

    distances[("check","sim_post_code")] =       postcode_compare(addr_matches[postcode_field_a].fillna(""), addr_matches[postcode_field_b].fillna(""))


    distances[("check","sim_city")] =      city_compare(addr_matches[city_field_a].fillna(""), addr_matches[city_field_b].fillna(""))

    elimination_rule = ((distances[("check","sim_post_code")] < 0.1) & (distances[("check","sim_city")] < similarity_threshold)) | (distances[("check","sim_street")] < similarity_threshold)



    rejected = addr_matches[elimination_rule].merge(distances, left_index=True, right_index=True).copy()

    rejected[("metadata","reject_reason")] = "mismatch"


    # Remove non acceptable results
    result = addr_matches[~elimination_rule].merge(distances, left_index=True, right_index=True).sort_values([addr_key_field, ("check","sim_street"), ("check","sim_house_number"), secondary_sort_field], ascending=[True, False, False, True])

    vlog("result:")
    vlog(result)
    # Keep only the first ones
    result_head = result.groupby([addr_key_field]).head(max_res)#.drop("level_2", axis=1)#.set_index([key, addr_key_field])#[init_osm.index.get_level_values(1) == 140266    ]

    result_tail = result[~result.index.isin(result_head.index)].copy()
    result_tail[("metadata", "reject_reason")] = "tail"

    keep = result_head
    reject = pd.concat([rejected, result_tail])
    return keep, reject





def match_parent(osm_results, osm_reject):
    """
    From records in osm_reject, check that using parent node, a record count
    become acceptable.

    Parameters
    ----------
    osm_results : pd.DataFrame
        First return value from osm_keep_relevant_results
    osm_reject : pd.DataFrame
        Second return value from osm_keep_relevant_results.

    Returns
    -------
    osm_results : pd.DataFrame
        Same as input, appended by records from osm_reject who became acceptable
        by checking parent.
    osm_reject : pd.DataFrame
        Same as in put, minus records who became acceptable by checking parent.

    """

    vlog("     - Trying alternative (parent) names for rejected answers")

    # Keep rejected records that do not correspond to an accepted address
    final_rejected = osm_reject[(osm_reject[("metadata", "reject_reason")] == "mismatch") &
                                (~osm_reject[addr_key_field].isin(osm_results[addr_key_field]))]


    final_rejected = final_rejected.drop(("metadata", "reject_reason"), axis=1)
    # Get parent place id from place id calling get_osm_details
    parent_place_id = final_rejected[("nominatim", "place_id")].apply(get_osm_details).apply(lambda x: (x["parent_place_id"] if "parent_place_id" in x else 0 ))

    if (parent_place_id == 0).any():
        log("Got some parent_place_id == 0")
        log(final_rejected[parent_place_id == 0])
        parent_place_id = parent_place_id[parent_place_id != 0]

    # Get alt names from details of parent
    alt_names =  parent_place_id.apply(get_osm_details).apply(lambda x: (x["names"], x["category"])).apply(pd.Series)

    # Keep only street parents, and split columns ("name", "name:"fr", "old_name" ...) in rows

    if alt_names.shape[0] >0 and alt_names[alt_names[1] == "highway"].shape[0] >0 :
        alt_names = alt_names[alt_names[1] == "highway"]


        alt_names = alt_names[0].apply(pd.Series).stack().reset_index(1)#.rename(columns= {0: ("nominatim", "alt_names")})


        alt_names.columns = pd.MultiIndex.from_tuples([("nominatim", "lang"),("nominatim", "alt_names")], names=["L0", "L1"])


        alt_names = final_rejected.merge(alt_names, left_index=True, right_index=True)


        # Keep only alt names that are different from street name
        alt_names = alt_names[alt_names[("output", "street_name")] != alt_names[("nominatim", "alt_names")]]

        # Remove "old" similarity values
        alt_names = alt_names.drop("check", axis=1, level=0).reset_index(drop=True)

        keep, _  = ignore_mismatch_keep_bests(alt_names,
                                      street_fields_a = [("nominatim", "alt_names")],
                                      housenbr_field_a = ("output", "house_number"),
                                      postcode_field_a = ("output","post_code"),
                                      city_field_a = ("output","post_name"),
                                      street_field_b = street_field,
                                      housenbr_field_b = housenbr_field,
                                      postcode_field_b = postcode_field,
                                      city_field_b = city_field)



        osm_results = pd.concat([osm_results, keep], sort=False)
    #     print(osm_reject.shape)
        osm_reject = osm_reject[~ osm_reject[[addr_key_field,("nominatim", "place_id")]].astype(str).apply(";".join, axis=1).isin(keep[[addr_key_field, ("nominatim", "place_id")]].astype(str).apply(";".join, axis=1)) ]
    #     print(osm_reject.shape)
        vlog("     - Saved : ")
        vlog(keep)
    else:
        vlog("     - Not any alt name")


    return osm_results, osm_reject



def osm_keep_relevant_results(osm_results, addresses, max_res=1):
    """

    For all records in osm_result (output from osm_parse_and_split ; several
    records could correspond to a single input address):
        - Check that all results are "close enough" from the input address
        - Keep in "keep", only one (valid) record for each input address, if any
        - Put in "reject" all other records, i.e., either records too far away
          from input address, or not the first result


    Parameters
    ----------
    osm_results : pd.DataFrame
        Output from osm_parse_and_split.
    addresses : pd.DataFrame
        Input addresses.
    max_res : int, optional
        How many result to keep per input address. The default is 1.

    Returns
    -------
    keep : pd.DataFrame
        Only one (valid) record for each input address, if any.
    reject : pd.DataFrame
        All other records, i.e., either records too far away from input address,
        or not the first result.

    """

    osm_results_street = osm_results.merge(addresses[[addr_key_field, street_field,#reset_index().
                                                                    postcode_field, housenbr_field,
                                                                    city_field, country_field]],
                                                         left_on=[addr_key_field],
                                                         right_on=[addr_key_field],
                                                         how="left").set_index(osm_results.index)

    assert osm_results_street.shape[0] == osm_results.shape[0]

    keep, reject  = ignore_mismatch_keep_bests(osm_results_street,
                                      street_fields_a = [("output", "street_name"),("output", "other"), ("nominatim","namedetails")],
                                      housenbr_field_a = ("output", "house_number"),
                                      postcode_field_a = ("output","post_code"),
                                      city_field_a = ("output","post_name"),
                                      street_field_b = street_field,
                                      housenbr_field_b = housenbr_field,
                                      postcode_field_b = postcode_field,
                                      city_field_b = city_field,
                                      secondary_sort_field = ("metadata", "osm_order"),
                                      max_res=max_res)

    return keep, reject
