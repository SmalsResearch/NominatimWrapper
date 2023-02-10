"""
Functions used in AddressCleanserREST

@author: Vandy Berten (vandy.berten@smals.be)
"""
#!/usr/bin/env python
# coding: utf-8

# pylint: disable=invalid-name
# pylint: disable=line-too-long


# In[37]:

import sys

from datetime import datetime, timedelta
import traceback

import re
import logging

from geopy.distance import distance

import pandas as pd
import numpy as np


# from tqdm.autonotebook import  tqdm

from config import (addr_key_field,
                    check_with_transformed,
                    with_cleansed_number_on_26,
                    use_osm_parent,
                    photon_check_results,
                    with_rest_libpostal,
                    street_field,
                    housenbr_field,
                    postcode_field,
                    city_field,
                    country_field,
                    regex_replacements,
                    collapse_params)

from base import (log, vlog, get_osm, get_osm_struct, get_photon)

from check_result_utils import (match_parent,
                                ignore_mismatch_keep_bests,
                                osm_keep_relevant_results
                                )


logging.basicConfig(format='[%(asctime)s]  %(message)s', stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)



if with_rest_libpostal:
    # Assuming LibpostalREST flask is running
    from base import parse_address
else:
    from postal.parser import parse_address

# tqdm.pandas()




timestats = {}


# # Functions

# ## Global


def update_timestats(label, start_time):
    """
    Add delay since 'start_time' (datetime.now()-start_time) to timestats[label]

    Parameters
    ----------
    label : str
        DESCRIPTION.
    t : timestamp
        when activity to measure started.

    Returns
    -------
    None.

    """
    if not label in timestats:
        timestats[label] = timedelta(0)
    timestats[label] += datetime.now() - start_time

def to_camel_case(data):
    """
    Convert a snake_case object to a camelCase.
    If d is a string, convert the string
    If d is a dict, convert all keys, recursively (i.e., values are dict or list), but not simple values
    If d is a list, convert all objects in the list

    Parameters
    ----------
    data: str, dict or list
        Object to camelize

    Returns
    -------
    Object of the same structure as data, but where :
    - dictionary keys have been camelized if data is a dict
    - input string has been camelized if data is a string

    """


    if isinstance(data, str):
        return re.sub(r"(_)([a-z])", lambda m: m.group(2).upper(),  data)
    if isinstance(data, dict):
        return { to_camel_case(key): to_camel_case(item) if isinstance(item, (dict, list)) else item for key, item in data.items()}
    if isinstance(data, list):
        return [ to_camel_case(item)  for item in data]
    return data

def retry_with_low_place_rank(osm_results, sent_addresses,
                              check_results=True, osm_structured=False):
    """
    For all OSM results in "osm_results", checks those with place_rank <0
    (i.e., those where building was not localized).
    For those, if they contain other signs that digits in the house number,
    try to clean the house number by removing any character after the first
    sequence of digits.
    Retry then to send this cleansed address to OSM. If this give a result with
    place_rank  == 30, keep it. Otherwise, keep the original value.

    Parameters
    ----------
    osm_results : pd.DataFrame
        Output of process_osm.
    sent_addresses : pd.DataFrame
        Output of the last transformer.
    check_results : bool, optional
        Parameter sent to process_osm. The default is True.
    osm_structured : bool, optional
        Parameter sent to process_osm. The default is False.

    Returns
    -------
    osm_results : pd.DataFrame
        input osm_result, with some potentiel improvements for results with
        place_rank < 30

    """

    start_time = datetime.now()
    vlog("Trying to improve place_rank with place_rank < 30 by cleansed house number ")
    sent_addresses_26 = osm_results[osm_results.place_rank < 30]

    if sent_addresses_26.shape[0]>0:
        sent_addresses_26 = sent_addresses_26.merge(sent_addresses)#[osm_addresses.place_rank == 26]

        vlog(f"    - <30: {sent_addresses_26.shape[0]}")
        sent_addresses_26 = sent_addresses_26[~sent_addresses_26[housenbr_field].fillna("").astype(str).str.match("^[0-9]*$")]
        vlog(f"    - numbers: {sent_addresses_26.shape[0]}")

        sent_addresses_26[housenbr_field] = sent_addresses_26[housenbr_field].fillna("").astype(str).str.extract("^([0-9]+)")[0]

        sent_addresses_26["osm_addr_in"] =   sent_addresses_26[street_field  ].fillna("") + ", "+\
                sent_addresses_26[housenbr_field].fillna("") +", " + \
                sent_addresses_26[postcode_field].fillna("") +" " +\
                sent_addresses_26[city_field    ].fillna("") +", "+\
                sent_addresses_26[country_field].fillna("")

        sent_addresses_26["osm_addr_in"]= sent_addresses_26["osm_addr_in"].apply(clean_addr_in)



        vlog(" ; ".join([f"rank {r}: {c}" for r, c in sent_addresses_26.place_rank.value_counts().iteritems()]))
        #print(osm_results_26.place_rank.value_counts())
        osm_results_26, _ = process_osm(sent_addresses_26,
                                                  osm_addr_field="osm_addr_in",
                                                  check_results=check_results,
                                                  osm_structured=osm_structured)

        if osm_results_26.shape[0]>0:
            pr_str=" ; ".join([f"rank {r}: {c}" for r, c in osm_results_26.place_rank.value_counts().iteritems()])
            vlog(f"     - New results with place_rank == 30 after cleansing ({pr_str}):")

            osm_results_26 = osm_results_26[osm_results_26.place_rank == 30]
            osm_results_26["retry_on_26"] = True

            osm_results = osm_results[~osm_results[addr_key_field].isin(osm_results_26[addr_key_field])].append(osm_results_26, sort=False)

    update_timestats("t&p > process > retry_low_rank", start_time)

    return osm_results


def get_lpost_house_number(street):
    """
    Use libpostal to extract house_number and box number from street.

    Parameters
    ----------
    street : str
        street name.

    Returns
    -------
    [house number, box number].

    """

    try:
        lpost = parse_address(street)


        housenbr = ";".join([y for (y, x) in lpost if x=="house_number"])
        boxnbr = ";".join([y for (y, x) in lpost if x=="unit"])
    except Exception as exc:
        log(f"Error during processing of 'get_lpost_house_number': {exc}")
        housenbr = 'error during Libpostal processing'
        boxnbr   = 'error during Libpostal processing'

    return [housenbr, boxnbr]


def add_extra_house_number(osm_addresses, addresses):
    """
    For all addresses in "addresses" (and their corresponding OSM results in
    osm_addresses), call libpostal to extract housenumbers

    Parameters
    ----------
    osm_addresses : pd.DataFrame
        Output of process_osm.
    addresses : pd.DataFrame
        input addresses.

    Returns
    -------
    pd.DataFrame equivalent from osm_addresses, with 3 additional columns:
        in_house_nbr: housenumber in input data
        lpost_house_nbr: housenumber from Libpostal
        lpost_unit: box number from Libpostal


    """
    vlog("Start adding extra house number")

    start_time = datetime.now()

    if "addr_out_number" not in osm_addresses:
        return osm_addresses

    result = osm_addresses.merge(addresses)
    result["in_house_nbr"] = result[housenbr_field]

    lp = result.fillna("").apply(lambda row: get_lpost_house_number(f"{row[street_field]} {row[housenbr_field]}, {row[postcode_field]} {row[city_field]}".strip()), axis=1,  result_type ='expand')

    result[["lpost_house_nbr", "lpost_unit"]] = lp

    vlog("End of adding extra house number")
    update_timestats("extra_hn", start_time)

    return result[np.concatenate([osm_addresses.keys(), ["in_house_nbr", "lpost_house_nbr", "lpost_unit"]])]


def clean_addr_in(addr_in):
    """
    Clean string containing an adresse, by removing useless commas. Examples:
        - "Avenue Fonsny, , , , " becomes "Avenue Fonsny"
        - "Avenue Fonsny, , 1060, , " becomes "Avenue Fonsny, 1060"

    Parameters
    ----------
    addr_in : str
        Address to clean.

    Returns
    -------
    addr_in : str
        Cleansed address (without duplicated commas).

    """
    old_addr_in=""
    while addr_in!= old_addr_in:
        old_addr_in = addr_in
        addr_in = re.sub(",[ ]*,", ",", addr_in).strip()
        addr_in = re.sub(",$", "", addr_in)
        addr_in = re.sub("^,", "", addr_in)

    return addr_in


def transform_and_process(to_process_addresses, transformers,
                          check_results=True, osm_structured=False):
    """
    - Apply the sequence of transformers (using apply_transformers)
    - Call OSM on the transformed addresses and parse results (using process_osm)
    - If there are results with place_rank< 30 (and with_cleansed_number_on_26),
      try to reprocess them (using retry_with_low_place_rank)

    Parameters
    ----------
    to_process_addresses : pd.DataFrame
        DESCRIPTION.
    transformers : list
        sequence of transformer names.
    check_results : bool, optional
        Parameter sent to "process_osm". The default is True.
    osm_structured : bool, optional
        Parameter sent to "process_osm". The default is False.

    Returns
    -------
    osm_results: pd.DataFrame
         OSM results.
    rejected: pd.DataFrame
         rejected recorts from OSM.
    step_stats: dict
        statistics.
    """

    start_time = datetime.now()
    method = "+".join(transformers)


    if to_process_addresses.shape[0]==0 or to_process_addresses[addr_key_field].duplicated().sum() > 0:

        vlog("No more addresses!")
        step_stats = {"method": method, "todo":  0, "sent": 0, "match": 0, "match_26": 0,
                      "reject_rec" :0, "reject_addr": 0, "reject_mism": 0}
        return pd.DataFrame(columns=[addr_key_field]), pd.DataFrame(columns=[addr_key_field, "reject_reason"]), step_stats


    transformed_addresses = apply_transformers(to_process_addresses, transformers,
                                               check_results=check_results)


    if transformed_addresses.shape[0]==0:
        vlog("No more addresses for this transformers sequence!")
        step_stats = {"method": method, "todo":  0, "sent": 0, "match": 0, "match_26": 0, "reject_rec" :0, "reject_addr": 0, "reject_mism": 0}
        return pd.DataFrame(columns=[addr_key_field]), pd.DataFrame(columns=[addr_key_field, "reject_reason"]), step_stats

    transformed_addresses["osm_addr_in"] =   transformed_addresses[street_field  ].fillna("") + ", "+\
                                             transformed_addresses[housenbr_field].fillna("") + ", "+\
                                             transformed_addresses[postcode_field].fillna("") + " " +\
                                             transformed_addresses[city_field    ].fillna("") + ", "+\
                                             transformed_addresses[country_field ].fillna("")


    transformed_addresses["osm_addr_in"]= transformed_addresses["osm_addr_in"].apply(clean_addr_in)



    if check_with_transformed :
        sent_addresses = transformed_addresses
    else:
        sent_addresses = transformed_addresses[["osm_addr_in", addr_key_field]].merge(to_process_addresses, on=addr_key_field)

    vlog(f"Will process {sent_addresses.shape[0]} addresses for : transformers = {'+'.join(transformers)}")

    vlog(sent_addresses.head())
    vlog(sent_addresses.shape)

    update_timestats("t&p > transformer", start_time)

    start_time = datetime.now()
    osm_results, rejected = process_osm(sent_addresses,
                                        osm_addr_field="osm_addr_in",
                                        check_results=check_results,
                                        osm_structured=osm_structured)

    if with_cleansed_number_on_26 and osm_results.shape[0]>0 :

        osm_results = retry_with_low_place_rank(osm_results, sent_addresses,
                                                check_results=check_results)

    update_timestats("t&p > process", start_time)
    osm_results["method"] = method
    rejected["method"] = method

    step_stats = {"method": method,
      "todo":        to_process_addresses.shape[0],
      "sent":        sent_addresses.shape[0],
      "match":       osm_results.shape[0],
      "match_26":    osm_results["retry_on_26"].sum() if "retry_on_26" in osm_results else 0,
      "reject_rec" : rejected.shape[0],
      "reject_addr": rejected[addr_key_field].nunique(),
      "reject_mism": rejected[rejected.reject_reason == "mismatch"][addr_key_field].nunique(),
     }

    return osm_results, rejected, step_stats


# ## OSM

def process_osm(df, osm_addr_field, accept_language="",
               check_results=True, osm_structured=False) :
    """
    Call OSM (Nominatim) for all records of df, sending field "osm_addr_field"
    (if osm_structured == False) to "get_osm" or [street_field, housenbr_field,
    postcode_field, city_field, country_field] (if osm_structured == True) to
    get_osm_struct.

    Parse then OSM results using "osm_parse_and_split".

    If "check_results" is True, eliminate too far appart results (from input)
    using osm_keep_relevant_results.

    For rejected results, if "use_osm_parent" is True (see config.py), see if,
    using "parent" record for a result, result was acceptable.

    ""

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with addresses to process.
    osm_addr_field : str
        Field containing address to send to OSM (if check_result).
    accept_language : str, optional
        Sent to get_osm/get_osm_struct. The default is "".
    check_results : bool, optional
        Should we trust OSM or not. The default is True.
    osm_structured : TYPE, optional
        Should we call the structured version of OSM (Nominatim) or not.
        The default is False.

    Returns
    -------
    Two dataframes:
        - One with accepted addresses (only one per input address)
        - One with rejected addresses, with
            - "rejected_reason" == "tail" if they were in the OSM results, but
              were not the first result
            - "rejected_reason" == "reject" if the were rejected as too far
              appart from input
    """

    start_time = datetime.now()

    if df.shape[0] == 0:
        return pd.DataFrame(columns=[osm_addr_field, addr_key_field]), \
                pd.DataFrame(columns=[osm_addr_field, addr_key_field, "reject_reason"])

    if osm_structured:
        to_process = df[[osm_addr_field, addr_key_field, street_field, housenbr_field,
                postcode_field, city_field, country_field]].drop_duplicates()
    else:
        to_process = df[[osm_addr_field]].drop_duplicates()

    vlog(f"OSM: Will process {df.shape[0]} with {to_process.shape[0]} unique values")

    vlog("Most frequent addresses:")
    vlog(df[osm_addr_field].value_counts().head(5))

    osm_res_field = "osm_res"

    to_process["accept_language"] = accept_language

    if osm_structured:
        to_process[osm_res_field] = to_process.apply(lambda row:
                                                   get_osm_struct(street =    row[street_field],
                                                                  housenumber=row[housenbr_field],
                                                                  postcode=   row[postcode_field],
                                                                  city=       row[city_field],
                                                                  country=    row[country_field],
                                                                  accept_language = row["accept_language"]), axis=1)
    else:
        to_process[osm_res_field] = to_process[[osm_addr_field, "accept_language"]].apply(lambda row: get_osm(row[osm_addr_field], row["accept_language"]), axis=1)

    update_timestats("t&p > process > osm", start_time)

    start_time = datetime.now()
    vlog("     - Parse & split osm results ...")

    osm_results = osm_parse_and_split(to_process,
                                      osm_res_field,
                                      osm_addr_field=osm_addr_field)

    to_process = None # Allows Garbage collector to free memory ...

    osm_results = df[[osm_addr_field, addr_key_field]].merge(osm_results)

    vlog(f"     - OSM got {osm_results.shape[0]} results for {osm_results[addr_key_field].nunique()} addresses")

    update_timestats("t&p > process > osm_post", start_time)

    if osm_results.shape[0] == 0:

        return osm_results, pd.DataFrame(columns=[osm_addr_field, addr_key_field, "reject_reason"])

    if check_results:

        start_time = datetime.now()

        vlog("     - Keep relevant results")
        osm_results, osm_reject = osm_keep_relevant_results(osm_results, df)


        if use_osm_parent:
            osm_results, osm_reject = match_parent(osm_results, osm_reject)
        vlog(f"     - Got {osm_results.shape[0]} results")



        update_timestats("t&p > process > checker", start_time)

    else:
        vlog("    - Do not check OSM results, just keep the first result for each request.")
        result_head = osm_results.groupby([addr_key_field]).head(1).copy()
        result_head["SIM_street_which"] = np.NaN

        osm_reject = osm_results[~osm_results.index.isin(result_head.index)].copy()
        osm_reject["SIM_street"]= np.NaN
        osm_reject["SIM_zip"]= np.NaN
        osm_reject["reject_reason"] = "tail"
        osm_results = result_head

    vlog("     - Done!")


    res_columns = [osm_addr_field, addr_key_field, "place_id", "lat", "lon", "display_name", "namedetails", "place_rank", "category", "type", "SIM_street_which", "SIM_street", "SIM_city", "SIM_zip", "SIM_house_nbr"] + list(collapse_params.keys()) + ["addr_out_other"]
    res_columns = [c for c in res_columns if c in osm_results ]
    return osm_results[res_columns], osm_reject

def osm_parse_and_split(df, osm_res_field,
                        osm_addr_field,
                        prefix="addr_",
                        drop_osm=True):
    """
    Reformat df, with one record per input address, and the whole OSM result
    in one cell, in a Dataframe, with one record for each result from OSM, and
    a set of additional columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    osm_res_field : str
        Column name of df containing OSM results.
    osm_addr_field : str
        Column name of df containing input address.
    prefix : str, optional
        Prefix to add to each output column. The default is "addr_".
    drop_osm : boolean, optional
        Should we drop column with OSM result (non parsed). The default is True.

    Returns
    -------
    pd.DataFrame
        DESCRIPTION.

    """

    osm_result_item_field = "osm_item_result"
    start_time = datetime.now()

    df = df.set_index([osm_addr_field])

    vlog("        * Unstacking...")
    df_split = df[osm_res_field].apply(pd.Series)

    if df_split.shape[1] == 0: # None of the addresses were matched by Nominatim
        return pd.DataFrame(columns = [osm_addr_field])

    osm_results = pd.DataFrame(df_split.stack()).rename(columns = {0:osm_result_item_field})

    vlog("        * Extract items")

    for item in osm_results.iloc[0][osm_result_item_field].keys() :
        osm_results[item] = osm_results[osm_result_item_field].apply(lambda x, it=item: x[it] if it in x else None)

    addr_items = []

    for row in osm_results[osm_result_item_field].apply(lambda x: x["address"]):
        for addr_item in row.keys():
            addr_items.append(addr_item)

    addr_items = pd.Series(addr_items).value_counts().keys().values

    for addr_item in addr_items:
        osm_results[prefix+addr_item] = osm_results[osm_result_item_field].apply(lambda x, ad_it=addr_item: x["address"][ad_it] if ad_it in x["address"] else None)

    # Keep only "namedetails" if category == "highway"
    osm_results["namedetails"] = np.where(osm_results["category"] == "highway", osm_results["namedetails"].apply(lambda dct: " - ".join(dct.values())), "")

    osm_results = osm_results.drop(columns=["address"] + ([osm_result_item_field] if drop_osm else []))

    osm_results.place_rank=osm_results.place_rank.astype(int)

    osm_results = add_addr_out_columns(osm_results, prefix)

    osm_results = osm_results.reset_index().rename(columns={"level_1": "osm_order"})

    update_timestats("t&p > process > osm_post > parse&split", start_time)

    return osm_results


def collapse(df, columns, prefix=None, method="fillna"):
    """
    Collapse all columns in "columns" into a single Series.
    If method = fillna, keep the first non null value (row by row)
    If method = set, keep all of them, without duplicates, separated by " - "

    Parameters
    ----------
    df : pd.DataFrame
        Any dataframe.
    columns : list of str (column names)
        List of columns to compact into a single out.
    prefix : str, optional
        If given, will prefix all columns in "columns" by this value beforehand. The default is None.
    method : str, optional
        "set" (keep all values) or "fillna" (keep only the first one).
        The default is "fillna".


    Returns
    -------
    res : pd.Series
        Collapsed column.

    """

    assert method in ["fillna", "set"], "method argument should be either 'fillna' or 'set'"


    if prefix:
        columns = [prefix+col for col in columns]

    if method=="fillna":

        res = pd.Series(index = df.index)#[columns[0]]

        for col in columns:
            if col in df.keys():
                res = res.fillna(df[col])
    elif method== "set":
        res = df[columns].apply(lambda lst: {x for x in lst if not pd.isnull(x)}, axis=1).apply(" - ".join)

    return res


def add_addr_out_columns(osm_results, prefix):
    """
    Add address component columns, by collapsing columns, following "collapse_params",
    using "collapse" with "fillna" method
    All other columns are gathered in "addr_out_other", using "collapse" wit "set" method

    Parameters
    ----------
    osm_results : pd.DataFrame

    prefix : str
        DESCRIPTION.

    Returns
    -------
    osm_results : pd.DataFrame
        input with a set of additional columns.

    """
    other_columns = osm_results.keys()

    for out_column, in_columns in collapse_params.items():
        other_columns = [col for col in other_columns if col.replace(prefix, "") not in in_columns and col.startswith(prefix)]
    other_columns.remove('addr_country_code')

    if "addr_state" in other_columns:
        other_columns.remove('addr_state')

    for out_column, in_columns in collapse_params.items():
        osm_results[out_column] = collapse(osm_results, in_columns, "addr_", "fillna")

    osm_results["addr_out_other"] = collapse(osm_results, other_columns, "", "set") if len(other_columns)>0 else np.NaN

    return osm_results


# # Transformers


def apply_transformers(addresses, transformers, check_results):
    """
    Apply the sequence of transformers given in "transformers" to all addresses
    in "addresses"

    Parameters
    ----------
    addresses : pd.DataFrame
        Addresses to "clean" (tranform).
    transformers : list (of string)
        DESCRIPTION.
    check_results : boolean
        For libpostal and photon.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if transformers == ["orig"]:
        return addresses.copy()

    init_addresses = addresses.copy()
    transformed_addresses = addresses.copy()

    for transformer in transformers:
        # TODO : cache system to avoid recompuing libpostal/photon when already done.
        vlog(f"   transformer: {transformer}")

        if transformer == "orig":
            pass # Don't do anything, keep original values

        elif re.match(r"regex\[[a-z]+\]", transformer):
            grp = re.match(r"regex\[([a-z]+)\]", transformer)
            regex_key = grp.groups(0)[0]

            transformed_addresses =  regex_transformer(transformed_addresses,
                                                       regex_key = regex_key)

        elif transformer == "nonum":
            transformed_addresses[housenbr_field] = ""

        elif transformer == "nostreet":
            transformed_addresses[housenbr_field] = ""
            transformed_addresses[street_field] = ""

        elif transformer == "nozip":
            transformed_addresses[postcode_field] = ""

        elif transformer == "nocountry":
            transformed_addresses[country_field] = ""

        elif transformer == "libpostal":
            transformed_addresses = libpostal_transformer(transformed_addresses,
                                                          check_results)
        elif transformer == "photon":
            transformed_addresses = photon_transformer(transformed_addresses,
                                                       photon_check_results)
        else :
            assert False, f"Wrong transformer type : {transformer}"

        if transformed_addresses.shape[0]==0:
            vlog("No more addresses after transformers!")
            transformed_addresses[addr_key_field] = np.NaN
            break

    # Return only records that have been modified by transfomer sequence

    changed = pd.Series(index=transformed_addresses.index)
    changed[:] = False

    fields = [street_field, housenbr_field, city_field, postcode_field, country_field]

    init_addresses = transformed_addresses[[addr_key_field]].merge(init_addresses).set_index(transformed_addresses.index)


    for field in fields:
        if field in transformed_addresses:
            changed = changed | (init_addresses[field].fillna("").astype(str).str.lower() != transformed_addresses[field].fillna("").astype(str).str.lower())

    return transformed_addresses[changed].copy()

# ## Photon

photon_street_field   = "photon_street"
photon_name_field     = "photon_name" # Sometimes, streetname is put in "name" field (especially for request without house number)
photon_postcode_field = "photon_postcode"
photon_city_field     = "photon_city"
photon_country_field  = "photon_country"


def photon_keep_relevant_results(photon_results, addresses):
    """
    Select from Photon result only those "close enough" from input addresses

    Parameters
    ----------
    photon_results : pd.DataFrame
        output of process_photon.
    addresses : pd.DataFrame
        Input addresses.

    Returns
    -------
    pd.DataFrame
        Selection of photon_results with only valide records.

    """
    photon_ext = photon_results.merge(addresses[[addr_key_field,  street_field,
                                                 housenbr_field, postcode_field,
                                                 city_field, country_field]])
    if photon_ext.shape[0] == 0:
        return pd.DataFrame()

    photon_ext["fake_house_number"] = ""

    vlog("Will compare photon results: ")
    vlog(photon_ext)
    keep, _  = ignore_mismatch_keep_bests(photon_ext,
                                  street_fields_a  = [photon_street_field],
                                  housenbr_field_a = "fake_house_number",
                                  postcode_field_a = photon_postcode_field,
                                  city_field_a = photon_city_field,
                                  street_field_b =   street_field,
                                  housenbr_field_b = "fake_house_number",
                                  postcode_field_b =   postcode_field,
                                  city_field_b =  city_field,
                                  secondary_sort_field = "photon_order")
    return keep


def photon_parse_and_split(res, addr_field, photon_col):
    """
    Parse Photon output, and split multiple results in a several rows

    Parameters
    ----------
    res : pd.DataFrame
        Dataframe containing Photon output.
    addr_field : str
        Column name containing address sent to Photon.
    photon_col : str
        Column name containing Photon output.

    Returns
    -------
    pd.DataFrame
        Parsed version of input.

    """
    res["photon_parsed"] = res[photon_col].apply(lambda j:j["features"] if "features" in j else None)

    res = res.set_index([addr_field])

    ser = res.photon_parsed.apply(pd.Series)

    if ser.shape[0] == 0 or ser.shape[1] == 0:
        return pd.DataFrame(columns = [addr_field])

    photon_results = pd.DataFrame(ser.stack()).rename(columns = {0:photon_col})

    for item in photon_results[photon_col].apply(lambda x: x.keys())[0]:
        photon_results[item] = photon_results[photon_col].apply(lambda x, it=item: x[it] if it in x else None)

    addr_items = []

    for row in photon_results[photon_col].apply(lambda x: x["properties"]):
        for addr_item in row.keys():
            addr_items.append(addr_item)

    addr_items = pd.Series(addr_items).value_counts().iloc[0:30].keys().values

    prefix="photon_"
    for addr_item in addr_items:
        photon_results[prefix+addr_item] = photon_results[photon_col].apply(lambda x, ad_it=addr_item: x["properties"][ad_it] if ad_it in x["properties"] else None)

    for fld in [photon_street_field, photon_postcode_field, photon_city_field, photon_country_field]:
        if fld not in photon_results:
            vlog(f"Photon: adding field {fld}")
            photon_results[fld] = ""

    if photon_name_field in photon_results:
        photon_results[photon_street_field] = photon_results[photon_street_field].replace("", pd.NA).fillna(photon_results[photon_name_field])

    photon_results["lat"] = photon_results["geometry"].apply(lambda x: x["coordinates"][0])
    photon_results["lon"] = photon_results["geometry"].apply(lambda x: x["coordinates"][1])

    photon_results = photon_results.drop([photon_col, "geometry", "photon_extent", "type","properties", "photon_osm_id"], axis=1, errors="ignore").reset_index().rename(columns={"level_1": "photon_order"})#res #parse_and_split(res, osm_field, key=addr_field)
    return photon_results

def process_photon(df, addr_field):
    """
    Sent addresses to Photon (with get_photon), and parse results (photon_parse_and_split)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with addresses to send to Photon.
    addr_field : str
        Column of df containing address.

    Returns
    -------
    photon_results : TYPE
        DESCRIPTION.

    """

    photon_col = "photon"

    to_process = df[[addr_field]].drop_duplicates()

    vlog(f"Photon: Will process {df.shape[0]} with {to_process.shape[0]} unique values")

    to_process[photon_col] = to_process[addr_field].apply(get_photon)

    photon_results = photon_parse_and_split(to_process, addr_field, photon_col)

    vlog(f"Photon got {photon_results.shape[0]} results for {df.shape[0]} addresses")
    vlog(photon_results)

    photon_results = df[[addr_key_field, addr_field]].merge(photon_results)

    return photon_results


def photon_transformer(addresses, check_results):
    """
    Transform "addresses" using Photon

    Parameters
    ----------
    addresses : pd.DataFrame
        Addresses to transform.
    check_results : boolean
        Should we check Photon output.

    Returns
    -------
    pd.DataFrame
        Transformed version of "addresses.

    """
    start_time = datetime.now()
    photon_addr = addresses[[addr_key_field, street_field, housenbr_field,
                             postcode_field, city_field, country_field]].copy()

    photon_addr["photon_full_addr"] = photon_addr[street_field].fillna("") +", "+ photon_addr[postcode_field].fillna("") + " " +photon_addr[city_field].fillna("")+", "+                                 photon_addr[country_field].fillna("")

    # Send to Photon
    photon_res = process_photon(photon_addr, "photon_full_addr")

    if check_results : #and photon_check_results:

        photon_res_sel = photon_keep_relevant_results(photon_res, photon_addr)
    else:
        photon_res_sel = photon_res.merge(addresses[[addr_key_field, street_field,
                                                     housenbr_field, postcode_field,
                                                     city_field, country_field]])

    if photon_res_sel.shape[0] == 0:
        return photon_res_sel

    fields = [(street_field, photon_street_field),
              (housenbr_field, housenbr_field), # We do not consider photon house number
              (city_field, photon_city_field),
              (postcode_field, photon_postcode_field),
              (country_field, photon_country_field)]

    fields_out    = [field_in      for field_in, field_photon in fields]
    fields_photon = [field_photon  for field_in, field_photon in fields]

    update_timestats("'t&p > transformer > photon", start_time)

    return photon_res_sel[[addr_key_field] + fields_photon].rename(columns= {field_photon: field_in for field_in, field_photon in fields})[[addr_key_field] + fields_out]


# ## Libpostal

lpost_street_field   = "lpost_road"
lpost_housenbr_field = "lpost_house_number"
lpost_postcode_field = "lpost_postcode"
lpost_city_field     = "lpost_city"
lpost_country_field  = "lpost_country"

def libpostal_transformer(addresses,
                          check_results):
    """
    Transform "addresses" using libpostal

    Parameters
    ----------
    addresses : pd.DataFrame
        Addresses to transform.
    check_results : boolean
        Should we check libpostaloutput.

     Returns
     -------
     pd.DataFrame
         Transformed version of "addresses.
    """
    start_time = datetime.now()

    libpost_addr = addresses[[addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field]].copy()

    # Make full address for libpostal
    libpost_addr["lpost_full_addr_in"] = libpost_addr[street_field] + ", "+ libpost_addr[housenbr_field].fillna("")+", "+                    libpost_addr[postcode_field].fillna("") + " " +libpost_addr[city_field].fillna("") +",  " +                    libpost_addr[country_field].fillna("")

    # Apply libpostal
    libpost_addr["lpost"] = libpost_addr.lpost_full_addr_in.apply(parse_address)
    libpost_addr["lpost"] = libpost_addr.lpost.apply(lambda lst: {x: y for (y, x) in lst})

    # Split libpostal results
    for field in "road", "house_number", "postcode", "city", "house", "country":
        libpost_addr["lpost_"+field] =libpost_addr.lpost.apply(lambda rec, fld=field: rec[fld] if fld in rec else np.NAN)

    if check_results:
        # Keep only "close" results
        libpost_addr, reject  = ignore_mismatch_keep_bests(libpost_addr,
                                      street_fields_a      = [street_field],
                                      housenbr_field_a     = housenbr_field,
                                      postcode_field_a     = postcode_field,
                                      city_field_a         = city_field,
                                      street_field_b       = lpost_street_field,
                                      housenbr_field_b     = lpost_housenbr_field,
                                      postcode_field_b     = lpost_postcode_field,
                                      city_field_b         = lpost_city_field,
                                      secondary_sort_field = addr_key_field)
        vlog("Rejected lipbostal results: ")
        vlog(reject)
    if libpost_addr.shape[0] == 0:

        return pd.DataFrame(columns=["osm_addr_in", addr_key_field])#,  libpost_addr


    fields =        [(street_field, lpost_street_field), (housenbr_field, lpost_housenbr_field),
                     (city_field, lpost_city_field), (postcode_field, lpost_postcode_field),
                     (country_field, lpost_country_field) ]
    fields_out    = [field_in      for field_in, field_lpost in fields]
    fields_lpost  = [field_lpost   for field_in, field_lpost in fields]

    update_timestats("'t&p > transformer > libpostal", start_time)

    return libpost_addr[[addr_key_field] + fields_lpost].rename(columns= {field_lpost: field_in for field_in, field_lpost in fields})[[addr_key_field] + fields_out]

# ## Regex transformer

def regex_transformer(addresses, regex_key="init"):
    """
    Transform "addresses" applying regex defined in config.regex_replacements

    Parameters
    ----------

    addresses : pd.DataFrame
        Addresses to transform.
    regex_key : str, optional
        set of regex to consider in config.regex_replacements. The default is "init".

    Returns
    -------
    pd.DataFrame
        Transformed version of addresses.

    """

    regex_addr = addresses[[addr_key_field, street_field, housenbr_field,
                            postcode_field, city_field, country_field]].copy()

    for (field, match, repl) in regex_replacements[regex_key]:
        vlog(f"{field}: {match}")
        new_values = regex_addr[field].fillna("").str.replace(match, repl)
        new_values_sel = regex_addr[field].fillna("") != new_values

        if new_values_sel.sum()>0:
            vlog(regex_addr[new_values_sel])

            regex_addr[field] = new_values
            vlog("-->")
            vlog(regex_addr[new_values_sel])
        else:
            vlog("None")

    return regex_addr

## REST utils


def get_init_df(data):
    """
    Build a single-record dataframe with data in data

    Parameters
    ----------
    data : dict
        Address components.

    Returns
    -------
    pd.DataFrame
        Single-record dataframe ready for "batch" processing.

    """
    log("init_df:")
    log(data)
    return pd.DataFrame([{addr_key_field : "1",
                          street_field:   data[street_field],
                          housenbr_field: data[housenbr_field],
                          postcode_field: data[postcode_field],
                          city_field:     data[city_field],
                          country_field:  data[country_field]
                          }])



def get_row_dict(row):
    """

    Convert a dataframe row in a dictionary (for a selection of columns)

    Parameters
    ----------
    row : pd.Series
        Dataframe row.

    Returns
    -------
    dict
        .

    """

    to_copy_field = ["osm_id", "place_id", "lat","lon","display_name",
                     "place_rank", "method",
                     "extra_house_nbr", "in_house_nbr", "lpost_house_nbr", "lpost_unit",
                     "reject_reason", "osm_addr_in"] + \
            list(collapse_params.keys())  + \
            list(filter(lambda x: x.startswith("SIM"), row.index))
    res =  {}

    for fld in to_copy_field:
        if fld in row:
            res[fld] = row[fld]

    return res


def format_res(res):
    """
    Convert a dataframe in a list of dictionaries

    Parameters
    ----------
    res : pd.DataFrame
        Dataframe to convert.

    Returns
    -------
    list of dict
        .

    """
    return list(res.fillna("").apply(get_row_dict, axis=1))



def format_osm_addr(osm_rec):
    """
    Format OSM output (only for "fast" method), using "collapse_params" config

    Parameters
    ----------
    osm_rec : dict
        OSM output.

    Returns
    -------
    res : dict
        Compacted/formatted version of OSM output.

    """
    res = {"method":"fast"}

    for fld in ["display_name", "place_id", "lat","lon", "place_rank"]:
        if fld in osm_rec:
            res[fld] = osm_rec[fld]

    for out_field, in_fields in collapse_params.items():
        res[out_field] = ""
        for in_field in in_fields:
            if in_field in osm_rec["address"]:
                res[out_field] = osm_rec["address"][in_field]
                break
    return res


def add_lpost_house_number(addr_in, match, data):
    """
    Use libpostal to get housenumber from address (only for "fast" method)

    Parameters
    ----------
    addr_in : str
        address to parse.
    match : dict
        OSM output.
    data : dict
        intput address.

    Returns
    -------
    None.

    """
    lpost = get_lpost_house_number(addr_in)
    match["in_house_nbr"] = data["housenumber"] if "housenumber" in data else ""
    match["lpost_house_nbr"] = lpost[0]
    match["lpost_unit"] = lpost[1]


def process_address_fast(data, osm_structured=False,
                         with_extra_house_number=True,
                         retry_with_low_rank=True):
    """
    Fast method, allowing to bypass all transformers/bypass procedure

    Parameters
    ----------
    data : dict
        address to geocode.
    osm_structured : bool, optional
        Should we used structured version of OSM. The default is False.
    with_extra_house_number : bool, optional
        Should we call libpostal to extract housenumber. The default is True.
    retry_with_low_rank : boolean, optional
        If result has a place_rank < 30, try to clean housenumber and start again.
        The default is True.

    Returns
    -------
    dict or None
        If success, dictionary with result. None otherwise.

    """
    start_time = datetime.now()


    addr_in = f"{data[street_field]}, {data[housenbr_field]}, {data[postcode_field]} {data[city_field]}, {data[country_field]}"

    addr_in= clean_addr_in(addr_in)

    try:

        if osm_structured:
            osm_res = get_osm_struct(street=     data[street_field],
                                    housenumber=data[housenbr_field],
                                    postcode=   data[postcode_field],
                                    city=       data[city_field],
                                    country=    data[country_field]
                                   )
        else:
            osm_res = get_osm(addr_in)
    except Exception as exc:
        log(f"Error during Nominatim call : {exc}")
        vlog(traceback.format_exc())
        return {"error": f"Error during Nominatim call: {exc}"}

    update_timestats("fast > osm", start_time)

    if len(osm_res) >0:
        match = format_osm_addr(osm_res[0])

        if retry_with_low_rank and match["place_rank"] < 30: # Try to clean housenumber to see if we can improved placerank
            vlog("Trying retry_with_low_rank")
            start_timet2 = datetime.now()
            cleansed_housenbr = re.match("^([0-9]+)", data[housenbr_field])
            #vlog(f"cleansed_housenbr: {cleansed_housenbr}")
            if cleansed_housenbr:
                cleansed_housenbr = cleansed_housenbr[0]

            #vlog(f"cleansed_housenbr: {cleansed_housenbr}")
            if cleansed_housenbr != data[housenbr_field]:

                data_cleansed= data.copy()
                data_cleansed[housenbr_field] = cleansed_housenbr
                osm_res_retry = process_address_fast(data_cleansed,
                                                     osm_structured=osm_structured,
                                                     with_extra_house_number=False,
                                                     retry_with_low_rank = False)

                if osm_res_retry and 'error' in osm_res_retry:
                    return osm_res_retry

                if osm_res_retry and osm_res_retry["match"][0]["place_rank"] == 30: # if place_rank is not improved, we keep the original result
                    osm_res_retry["match"][0]["cleansed_house_nbr"] = cleansed_housenbr
                    if with_extra_house_number:
                        add_lpost_house_number(addr_in, osm_res_retry["match"][0], data)

                    update_timestats("fast > retry", start_timet2)
                    return osm_res_retry

        if with_extra_house_number:
            add_lpost_house_number(addr_in, match, data)

        start_time2 = datetime.now()
        match["osm_addr_in"] = addr_in
        res = {"match":  [match],
               "reject": []}

        for osm_rec in osm_res[1:]:
            rec = format_osm_addr(osm_rec)
            rec["reject_reason"]= "tail"
            rec["dist_to_match"] = round(distance( (rec["lat"], rec["lon"]), (match["lat"], match["lon"])).km, 3)

            res["reject"].append(rec)

        update_timestats("fast > format", start_time2)
        update_timestats("fast", start_time)
        return res

    update_timestats("fast", start_time)
    return None



def process_address(data, check_results=True,
                    osm_structured=False,
                    with_extra_house_number=True,
                    fastmode=True,
                    transformers_sequence=None):
    """
    Main logical for single address

    Parameters
    ----------
    data : dict
        address to geocode.
    check_results : boolean, optional
        Should we check OSM output. The default is True.
    osm_structured : boolean, optional
        Should we use structred version of OSM. The default is False.
    with_extra_house_number : boolean, optional
        Should we use libpostal to extract housenumber. The default is True.
    fastmode : boolean, optional
        Should we first try "fast" method. The default is True.
    transformers_sequence : list (of list of str), optional
        List of sequences of transformers. The default is None.

    Returns
    -------
    dict
        geocoding result.

    """

    vlog(f"Will process {data}")
    #start_time = datetime.now()

    if transformers_sequence is None:
        transformers_sequence  = [["orig"]]

    if fastmode and not check_results:
        vlog("Try fast mode")
        res = process_address_fast(data, osm_structured=osm_structured, with_extra_house_number=with_extra_house_number)
        if res:
            return res
        vlog("No result in fast mode, go to full batch mode")

    to_process_addresses = get_init_df(data)

    vlog("Got dataframe")
    all_reject = pd.DataFrame()
    for transformers in (transformers_sequence if (not fastmode or check_results) else transformers_sequence[1:] ) : # Assumes ['orig'] is always the first transf. sequence
        vlog ("--------------------------")
        vlog("| Transformers : " + ";".join(transformers))
        vlog ("--------------------------")

        try :
            start_time = datetime.now()
            osm_results, rejected, step_stats = transform_and_process(to_process_addresses, transformers,
                                                                      check_results=check_results,
                                                                      osm_structured=osm_structured)
            update_timestats("t&p", start_time)
        except Exception as exc:
            log(f"Error during processing : {exc}")
            vlog(traceback.format_exc())
            return {"error": str(exc)}

        all_reject = all_reject.append(rejected, sort=False)

        vlog(step_stats)
        if osm_results.shape[0] > 0:
            if with_extra_house_number :
                osm_results = add_extra_house_number(osm_results, to_process_addresses)

            start_time = datetime.now()
            form_res =  format_res(osm_results)
            form_rej = format_res(all_reject)
            update_timestats("format_res", start_time)

            return {"match": form_res, "reject": form_rej }

    return {"reject": format_res(all_reject)}


def process_addresses(to_process_addresses, check_results=True,
                      osm_structured=False,
                      with_extra_house_number=True,
                      transformers_sequence=None
                      ):
    """
    Main logical for batch addresses

    Parameters
    ----------
    to_process_addresses : pd.DataFrame
        addresses to process.
    check_results : boolean, optional
        Should we check OSM output. The default is True.
    osm_structured : boolean, optional
        Should we use structred version of OSM. The default is False.
    with_extra_house_number : boolean, optional
        Should we use libpostal to extract housenumber. The default is True.
    transformers_sequence : list (of list of str), optional
        List of sequences of transformers. The default is None.

    Returns
    -------
    osm_addresses : pd.DataFrame
        geocoding result.
    rejected_addresses : pd.DataFrame
        Rejected addresses.

    """

    if transformers_sequence is None:
        transformers_sequence  = [["orig"]]

    osm_addresses        = pd.DataFrame()
    rejected_addresses   = pd.DataFrame()

    chunk = to_process_addresses.copy()

    for transformers in transformers_sequence:
        vlog ("--------------------------")
        log(f"Transformers ({chunk.shape[0]:3} records): " + ";".join(transformers))
        vlog ("--------------------------")

        try :
            osm_results, rejected, step_stats = transform_and_process(chunk, transformers,
                                                                      check_results=check_results,
                                                                      osm_structured=osm_structured)

            osm_addresses =      osm_addresses.append(osm_results, sort=False).drop_duplicates()
            rejected_addresses = rejected_addresses.append(rejected, sort=False).drop_duplicates()

        except Exception as exc:
            osm_results = chunk[[addr_key_field]]
            osm_results["method"] = "error on " + ";".join(transformers)
            osm_addresses =      osm_addresses.append(osm_results, sort=False).drop_duplicates()

            log(f"Error during processing: {exc}")
            vlog(traceback.format_exc())

        chunk  = chunk[~chunk[addr_key_field].isin(osm_results[addr_key_field])].copy()
        if chunk.shape[0]==0:
            break


        vlog(step_stats)
    if with_extra_house_number and osm_addresses.shape[0] > 0:
        osm_addresses = add_extra_house_number(osm_addresses, to_process_addresses)

    return osm_addresses, rejected_addresses #{"match": format_res(osm_results), "rejected": format_res(all_reject)}






def remove_empty_values(dct_lst):
    """
    Remove empty values in a list of dict

    Parameters
    ----------
    dct_lst : list (of dicts)
        List of dictionaries.

    Returns
    -------
    list
        Copy of input list, but all empty values in dicts are dropped

    """

    return [{k: v for k, v in item.items() if not pd.isnull(v) and v != ""} for item in dct_lst]


def convert_street_components(osm_record):
    """
    Convert a record containing fields like "addr_out_street", "addr_out_number"
    to a dict containing a field "address":{"streetName":..., "houseNumber": ...}

    Parameters
    ----------
    osm_record : dict
        input OSM record

    Returns
    -------
    osm_record : dict
        Idem as input, with an additional field "address", and without "addr_out_*" fields

    """

    address_out = {
        "streetName"  : osm_record["addr_out_street"]  if "addr_out_street"    in osm_record else None,
        "houseNumber" : osm_record["addr_out_number"]  if "addr_out_number"    in osm_record else None,
        "postCode"    : osm_record["addr_out_postcode"]if "addr_out_postcode"  in osm_record else None,
        "city"        : osm_record["addr_out_city"]    if "addr_out_city"      in osm_record else None,
        "country"     : osm_record["addr_out_country"] if "addr_out_country"   in osm_record else None,
        "other"       : osm_record["addr_out_other"]   if "addr_out_other"     in osm_record else None
    }
    osm_record["address"] = {k:v for k,v in address_out.items() if v is not None}

    addr_out_keys = list(filter(lambda k: "addr_out" in k, osm_record.keys()))
    for fld in addr_out_keys:
        del  osm_record[fld]
    return osm_record
