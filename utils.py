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

from datetime import datetime
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
                    transformed_address_field,
                    collapse_params)

from base import (log, vlog, get_osm, get_osm_struct, update_timestats)

from check_result_utils import (match_parent,
                                osm_keep_relevant_results
                                )

from transformers import (photon_transformer, libpostal_transformer, regex_transformer)


logging.basicConfig(format='[%(asctime)s]  %(message)s', stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)



if with_rest_libpostal:
    # Assuming LibpostalREST flask is running
    from base import parse_address
else:
    from postal.parser import parse_address

# tqdm.pandas()





###############################
## Data conversion
###############################

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

def collapse(df, columns, method="fillna"):
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
    method : str, optional
        "set" (keep all values) or "fillna" (keep only the first one).
        The default is "fillna".


    Returns
    -------
    res : pd.Series
        Collapsed column.

    """

    assert method in ["fillna", "set"], "method argument should be either 'fillna' or 'set'"


    if method=="fillna":

        res = pd.Series(index = df.index, dtype=str)#[columns[0]]

        for col in columns:
            if col in df.keys():
                res = res.fillna(df[col])
    elif method== "set":
        res = df[columns].apply(lambda lst: {x for x in lst if not pd.isnull(x)}, axis=1).apply(" - ".join)

    return res



def multiindex_to_dict(df):
    """
    Convert a Pandas Dataframe with two-levels multi-index columns into
    a list of dictionaries

    Example of input:
    L0    | col_1   | col_1   | col_2    |
    L1    | col_1_1 | col_1_2 | col_2_1  |
    --------------------------------------
    0     | A       | B       | C        |
    1     | D       | E       | F        |
    --------------------------------------

    corresponding output:
        [ {'col_1' : {
                'col_1_1':'A',
                'col_1_2':'B'},
            'col_2' : {
                'col_2_1': 'C'
                }},
        {'col_1' : {
                'col_1_1':'D',
                'col_1_2':'E'},
            'col_2' : {
                'col_2_1': 'F'
                }}
            ]

    Parameters
    ----------
    df : pd.DataFrame
        a Pandas Dataframe with two-levels multi-index columns.

    Returns
    -------
    list of dict


    """
    l0_names = df.columns.get_level_values(0).unique()

    res =  [{ k1: {k2: rec[(k1, k2)] for k2 in df[k1].columns.get_level_values(0)} for k1 in l0_names }  for rec in df.to_dict(orient="records")]


    return [{k1: {k2: rec[k1][k2] for k2 in rec[k1] if isinstance(rec[k1][k2], list) or (pd.notnull(rec[k1][k2]) and rec[k1][k2] != "")} for k1 in rec} for rec in res]


###############################
## OSM functions
###############################

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
    sent_addresses_26 = osm_results[osm_results[("nominatim", "place_rank")] < 30]

    if sent_addresses_26.shape[0]>0:
        sent_addresses_26 = sent_addresses_26.merge(sent_addresses)#[osm_addresses.place_rank == 26]

        vlog(f"    - <30: {sent_addresses_26.shape[0]}")
        sent_addresses_26 = sent_addresses_26[~sent_addresses_26[housenbr_field].fillna("").astype(str).str.match("^[0-9]*$")]
        vlog(f"    - numbers: {sent_addresses_26.shape[0]}")

        sent_addresses_26 = sent_addresses_26.fillna("").astype(str)
        sent_addresses_26[housenbr_field] = sent_addresses_26[housenbr_field].str.extract("^([0-9]+)")[0]

        sent_addresses_26[transformed_address_field] =   sent_addresses_26[street_field  ].fillna("") + ", "+\
                sent_addresses_26[housenbr_field].fillna("") +", " + \
                sent_addresses_26[postcode_field].fillna("") +" " +\
                sent_addresses_26[city_field    ].fillna("") +", "+\
                sent_addresses_26[country_field].fillna("")

        sent_addresses_26[transformed_address_field]= sent_addresses_26[transformed_address_field].apply(clean_addr_in)



        vlog(" ; ".join([f"rank {r}: {c}" for r, c in sent_addresses_26[("nominatim", "place_rank")].value_counts().items()]))
        #print(osm_results_26.place_rank.value_counts())
        osm_results_26, _ = process_osm(sent_addresses_26,
                                                  osm_addr_field=transformed_address_field,
                                                  check_results=check_results,
                                                  osm_structured=osm_structured)

        if osm_results_26.shape[0]>0:
            pr_str=" ; ".join([f"rank {r}: {c}" for r, c in osm_results_26[("nominatim", "place_rank")].value_counts().items()])
            vlog(f"     - New results with place_rank == 30 after cleansing ({pr_str}):")

            osm_results_26 = osm_results_26[osm_results_26[("nominatim", "place_rank")] == 30]
            osm_results_26[("work", "retry_on_26")] = True

            osm_results = pd.concat([osm_results[~osm_results[addr_key_field].isin(osm_results_26[addr_key_field])],
                                     osm_results_26], sort=False)

    update_timestats("t&p > process > retry_low_rank", start_time)

    return osm_results



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
            - "rejected_reason" == "mismatch" if the were rejected as too far
              appart from input
    """

    start_time = datetime.now()

    if df.shape[0] == 0:
        return pd.DataFrame(columns=[osm_addr_field, addr_key_field]), \
                pd.DataFrame(columns=[osm_addr_field, addr_key_field, ("work", "reject_reason")])

    if osm_structured:
        to_process = df[[osm_addr_field, addr_key_field, street_field, housenbr_field,
                postcode_field, city_field, country_field]].drop_duplicates()
    else:
        to_process = df[[osm_addr_field]].drop_duplicates()

    vlog(f"OSM: Will process {df.shape[0]} with {to_process.shape[0]} unique values")

    vlog("Most frequent addresses:")
    vlog(df[osm_addr_field].value_counts().head(5))

    osm_res_field = ("nominatim", "result")

    to_process[("work", "accept_language")] = accept_language

    if osm_structured:
        to_process[osm_res_field] = to_process.apply(lambda row:
                                                   get_osm_struct(street =    row[street_field],
                                                                  housenumber=row[housenbr_field],
                                                                  postcode=   row[postcode_field],
                                                                  city=       row[city_field],
                                                                  country=    row[country_field],
                                                                  accept_language = row[("work", "accept_language")]), axis=1)
    else:
        to_process[osm_res_field] = to_process[[osm_addr_field, ("work", "accept_language")]].apply(lambda row: get_osm(row[osm_addr_field], row[("work","accept_language")]), axis=1)

    update_timestats("t&p > process > osm", start_time)

    start_time = datetime.now()
    vlog("     - Parse & split osm results ...")

    osm_results = osm_parse_and_split(to_process,
                                      osm_res_field,
                                      osm_addr_field=osm_addr_field)

    to_process = None # Allows Garbage collector to free memory ...


    osm_results = df[[osm_addr_field, addr_key_field]].merge(osm_results)

    vlog(f"     - OSM got {osm_results.shape[0]} results for {osm_results[addr_key_field].nunique()} addresses")
    vlog(osm_results)

    update_timestats("t&p > process > osm_post", start_time)

    if osm_results.shape[0] == 0:

        return osm_results, pd.DataFrame(columns=[osm_addr_field, addr_key_field, ("work", "reject_reason")])

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

        osm_reject = osm_results[~osm_results.index.isin(result_head.index)].copy()
        
#         log("head:")
#         log(result_head)
#         log("rejected: ")
#         log(osm_reject)
        
#         osm_reject = osm_reject.merge(result_head[[addr_key_field, ("nominatim", "lat"), ("nominatim", "lon")]].rename(columns={"nominatim": "nominatim_match"}))
#         log(osm_reject)
        
        osm_reject[("work", "reject_reason")] = "tail"
        #osm_reject[("work", "dist_to_match")] = osm_reject.apply(lambda rec: round(distance( (rec[("nominatim", "lat")], rec[("nominatim", "lon")]), (rec[("nominatim_match", "lat")], rec[("nominatim_match", "lon")])).km, 3), axis=1)
        
        
        osm_results = result_head

    vlog("     - Done!")


    return osm_results, osm_reject

def osm_parse_and_split(df, osm_res_field,
                        osm_addr_field,
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
    drop_osm : boolean, optional
        Should we drop column with OSM result (non parsed). The default is True.

    Returns
    -------
    pd.DataFrame
        DESCRIPTION.

    """


    osm_result_item_field = ("nominatim", "osm_item_result")
    start_time = datetime.now()

    df = df.set_index([osm_addr_field])

    vlog("        * Unstacking...")
    df_split = df[osm_res_field].apply(pd.Series, dtype=object)



    if df_split.shape[1] == 0: # None of the addresses were matched by Nominatim
        log("No result")
        osm_results=pd.DataFrame(columns = [osm_addr_field])
        osm_results.columns = pd.MultiIndex.from_tuples(osm_results.columns, names=["L0", "L1"])
        osm_results.columns = osm_results.columns.set_names(names=["L0", "L1"])


        return osm_results

    osm_results = pd.DataFrame(df_split.stack()).rename(columns = {0:osm_result_item_field})

    osm_results.columns = pd.MultiIndex.from_tuples([osm_result_item_field], names=["L0", "L1"])

    vlog("        * Extract items")

    for item in osm_results.iloc[0][osm_result_item_field].keys() :
        osm_results[("nominatim", item)] = osm_results[osm_result_item_field].apply(lambda x, it=item: x[it] if it in x else None)

    addr_items = []

    for row in osm_results[osm_result_item_field].apply(lambda x: x["address"]):
        for addr_item in row.keys():
            addr_items.append(addr_item)

    addr_items = list(set(addr_items))


    for addr_item in addr_items:
        osm_results[("nominatim", addr_item)] = osm_results[osm_result_item_field].apply(lambda x, ad_it=addr_item: x["address"][ad_it] if ad_it in x["address"] else None)

    # log(osm_results)

    # Keep only "namedetails" if category == "highway"
    osm_results[("nominatim", "namedetails")] = np.where(osm_results[("nominatim", "category")] == "highway", osm_results[("nominatim", "namedetails")].apply(lambda dct: " - ".join(dct.values())), "")

    osm_results = osm_results.drop(columns=[("nominatim", "address")] + ([osm_result_item_field] if drop_osm else []))

    osm_results[("nominatim", "place_rank")]=osm_results[("nominatim", "place_rank")].astype(int)


    for fld in ["country_code","region"]:
        if fld in addr_items:
            addr_items.remove(fld)


    osm_results = add_addr_out_columns(osm_results,addr_items)

    osm_results = osm_results.reset_index()
    osm_results[("work", "osm_order")] = osm_results["level_1"]

    osm_results = osm_results.drop("level_1", level=0, axis=1)

    update_timestats("t&p > process > osm_post > parse&split", start_time)

    return osm_results



def add_addr_out_columns(osm_results, addr_items):
    """
    Add address component columns, by collapsing columns, following "collapse_params",
    using "collapse" with "fillna" method
    All other columns are gathered in "addr_out_other", using "collapse" with "set" method

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

    for out_column, in_columns in collapse_params.items():
        osm_results[out_column] = collapse(osm_results, [("nominatim", col) for col in in_columns], "fillna")


    osm_results[("output","other")] = osm_results["nominatim"][addr_items].apply(set,axis=1)


    for out_column, in_columns in collapse_params.items():
        osm_results[("output","other")] = osm_results.apply(lambda rec, oc=out_column: rec[("output","other")] -{rec[oc]}, axis=1)

    osm_results[("output","other")] = osm_results[("output","other")].apply(lambda lst: [x for x in lst if not pd.isnull(x)]).apply(" - ".join)

    return osm_results





##############################
## Global process
##############################

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
        return pd.DataFrame(columns=[addr_key_field]), pd.DataFrame(columns=[addr_key_field, ("work", "reject_reason")]), step_stats


    transformed_addresses = apply_transformers(to_process_addresses, transformers,
                                               check_results=check_results)


    if transformed_addresses.shape[0]==0:
        vlog("No more addresses for this transformers sequence!")
        step_stats = {"method": method, "todo":  0, "sent": 0, "match": 0, "match_26": 0, "reject_rec" :0, "reject_addr": 0, "reject_mism": 0}
        return pd.DataFrame(columns=[addr_key_field]), pd.DataFrame(columns=[addr_key_field, ("work", "reject_reason")]), step_stats

    transformed_addresses[transformed_address_field] =   transformed_addresses[street_field].fillna("") + ", "+\
                                             transformed_addresses[housenbr_field].fillna("") + ", "+\
                                             transformed_addresses[postcode_field].fillna("") + " " +\
                                             transformed_addresses[city_field   ].fillna("") + ", "+\
                                             transformed_addresses[country_field].fillna("")


    transformed_addresses[transformed_address_field]= transformed_addresses[transformed_address_field].apply(clean_addr_in)


    if check_with_transformed :
        sent_addresses = transformed_addresses
    else:
        sent_addresses = transformed_addresses[[transformed_address_field, addr_key_field]].merge(to_process_addresses, on=addr_key_field)

    vlog(f"Will process {sent_addresses.shape[0]} addresses for : transformers = {'+'.join(transformers)}")

    vlog(sent_addresses.head())
    vlog(sent_addresses.shape)

    update_timestats("t&p > transformer", start_time)

    start_time = datetime.now()
    osm_results, rejected = process_osm(sent_addresses,
                                        osm_addr_field=transformed_address_field,
                                        check_results=check_results,
                                        osm_structured=osm_structured)

    if with_cleansed_number_on_26 and osm_results.shape[0]>0 :

        osm_results = retry_with_low_place_rank(osm_results, sent_addresses,
                                                check_results=check_results)

    update_timestats("t&p > process", start_time)
    osm_results[("work", "method")] = method
    rejected[("work", "method")] = method

    #log(rejected)

    step_stats = {"method": method,
      "todo":        to_process_addresses.shape[0],
      "sent":        sent_addresses.shape[0],
      "match":       osm_results.shape[0],
      "match_26":    osm_results[("work", "retry_on_26")].sum() if ("work", "retry_on_26") in osm_results else 0,
      "reject_rec" : rejected.shape[0],
      "reject_addr": rejected[addr_key_field].nunique(),
      "reject_mism": rejected[rejected[("work", "reject_reason")] == "mismatch"][addr_key_field].nunique() if rejected.shape[0]>0 else 0,
     }

    return osm_results, rejected, step_stats



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

    changed = pd.Series(index=transformed_addresses.index, dtype=str)
    changed[:] = False

    fields = [street_field, housenbr_field, city_field, postcode_field, country_field]


    if transformed_addresses.shape[0]==0:
        transformed_addresses.columns = pd.MultiIndex.from_tuples(transformed_addresses.columns, names=["L0", "L1"])

    init_addresses = transformed_addresses[[addr_key_field]].merge(init_addresses).set_index(transformed_addresses.index)


    for field in fields:
        if field in transformed_addresses:
            changed = changed | (init_addresses[field].fillna("").astype(str).str.lower() != transformed_addresses[field].fillna("").astype(str).str.lower())

    return transformed_addresses[changed].copy()


##########################
## Libpostal functions
#########################

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


def add_extra_house_number(osm_addresses):
    """
    For all addresses in input fields of "osm_addresses", call libpostal to extract housenumbers

    Parameters
    ----------
    osm_addresses : pd.DataFrame
        Output of process_osm.

    Returns
    -------
    pd.DataFrame equivalent from osm_addresses, with 3 additional columns:
        in_house_number: housenumber in input data
        lpost_house_number: housenumber from Libpostal
        lpost_unit: box number from Libpostal


    """
    vlog("Start adding extra house number")

    start_time = datetime.now()

    result = osm_addresses

    result[("output", "in_house_number")] = result[housenbr_field]

    lp = result.fillna("").apply(lambda row: get_lpost_house_number(f"{row[street_field]} {row[housenbr_field]}, {row[postcode_field]} {row[city_field]}".strip()), axis=1,  result_type ='expand')

    result[[("output","lpost_house_number"), ("output","lpost_unit")]] = lp

    vlog("End of adding extra house number")
    update_timestats("extra_hn", start_time)

    return result




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

    match["output"]["in_house_number"] = data[housenbr_field] if housenbr_field in data else ""
    match["output"]["lpost_house_number"] = lpost[0]
    match["output"]["lpost_unit"] = lpost[1]


#######################
## REST utils
#######################


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
    vlog("init_df:")

    df= pd.DataFrame([{addr_key_field : data[addr_key_field] if addr_key_field in data and len(data[addr_key_field])>0 else "-1",
                          street_field:   data[street_field],
                          housenbr_field: data[housenbr_field],
                          postcode_field: data[postcode_field],
                          city_field:     data[city_field],
                          country_field:  data[country_field]
                          }])

    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["L0", "L1"])
    vlog(df)
    return df


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
    res = {"work": {"method":"fast"},
          "nominatim": {}}



    for fld in ["display_name", "place_id", "lat","lon", "place_rank"]:
        if fld in osm_rec:
            res["nominatim"][fld] = osm_rec[fld]
    for fld in osm_rec["address"]:
        res["nominatim"][fld] = osm_rec["address"][fld]


    for out_field, in_fields in collapse_params.items():
        if out_field[0] not in res:
            res[out_field[0]] = {}
        for in_field in in_fields:
            if in_field in osm_rec["address"]:
                res[out_field[0]][out_field[1]] = osm_rec["address"][in_field]
                break
    return res




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
        if retry_with_low_rank and match["nominatim"]["place_rank"] < 30: # Try to clean housenumber to see if we can improved placerank
            vlog("Trying retry_with_low_rank")
            start_timet2 = datetime.now()
            cleansed_housenbr = re.match("^([0-9]+)", data[housenbr_field])
            #vlog(f"cleansed_housenbr: {cleansed_housenbr}")
            if cleansed_housenbr:
                cleansed_housenbr = cleansed_housenbr[0]

            #vlog(f"cleansed_housenbr: {cleansed_housenbr}")
            if cleansed_housenbr and cleansed_housenbr != data[housenbr_field]:

                data_cleansed= data.copy()
                data_cleansed[housenbr_field] = cleansed_housenbr
                osm_res_retry = process_address_fast(data_cleansed,
                                                     osm_structured=osm_structured,
                                                     with_extra_house_number = False,
                                                     retry_with_low_rank = False)

                if osm_res_retry and 'error' in osm_res_retry:
                    return osm_res_retry

                if osm_res_retry and osm_res_retry["match"][0]["nominatim"]["place_rank"] == 30: # if place_rank is not improved, we keep the original result
                    osm_res_retry["match"][0]["work"]["cleansed_house_number"] = cleansed_housenbr
                    osm_res_retry["match"][0]["work"]["retry_on_26"] = True
                    if with_extra_house_number:
                        add_lpost_house_number(addr_in, osm_res_retry["match"][0], data)

                    update_timestats("fast > retry", start_timet2)
                    return osm_res_retry

        if with_extra_house_number:
            add_lpost_house_number(addr_in, match, data)

        start_time2 = datetime.now()
        match[transformed_address_field[0]][transformed_address_field[1]] = addr_in

        match[street_field[0]] = {}
        for f in street_field, housenbr_field, postcode_field, city_field, country_field, addr_key_field:
            match[f[0]][f[1]] = data[f]


        res = {"match":  [match],
               "rejected": []}

        for osm_rec in osm_res[1:]:
            rec = format_osm_addr(osm_rec)
            rec["work"]["reject_reason"]= "tail"
            rec["work"]["dist_to_match"] = round(distance( (rec["nominatim"]["lat"], rec["nominatim"]["lon"]), (match["nominatim"]["lat"], match["nominatim"]["lon"])).km, 3)

            res["rejected"].append(rec)

        update_timestats("fast > format", start_time2)
        update_timestats("fast", start_time)
        return res

    update_timestats("fast", start_time)
    return None



def add_dist_to_match(osm_results, osm_reject):
    if osm_reject.shape[0] ==0:
        return osm_reject
    # log("add_dist_to_match")
    s_bef =osm_reject.shape[0] 
    osm_reject = osm_reject.merge(osm_results[[addr_key_field, ("nominatim", "lat"), ("nominatim", "lon")]].rename(columns={"nominatim": "nominatim_match"}), how="left")
    
    
    # log(osm_reject)
    assert s_bef ==osm_reject.shape[0] 
    
    osm_reject[("work", "dist_to_match")] = osm_reject.apply(lambda rec: round(distance( (rec[("nominatim", "lat")], rec[("nominatim", "lon")]), (rec[("nominatim_match", "lat")], rec[("nominatim_match", "lon")])).km, 3), axis=1)
    
    return osm_reject.drop("nominatim_match", level=0, axis=1)
    
    
    
    
    
    
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

        all_reject = pd.concat([all_reject, rejected], sort=False)
        all_reject.columns = pd.MultiIndex.from_tuples(all_reject.columns, names=["L0", "L1"])

        vlog(step_stats)
        if osm_results.shape[0] > 0:

            osm_results = osm_results.drop([street_field, housenbr_field, postcode_field, city_field, country_field],
                                           errors='ignore',
                                           axis=1).merge(to_process_addresses, how="left")

            if with_extra_house_number :
                osm_results = add_extra_house_number(osm_results)
            
            all_reject = add_dist_to_match(osm_results, all_reject)
            
            start_time = datetime.now()
            form_res =  multiindex_to_dict(osm_results)


            form_rej = multiindex_to_dict(all_reject)
            #update_timestats("format_res", start_time)

            return {"match": form_res, "rejected": form_rej }

    return {"rejected": multiindex_to_dict(all_reject)}


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

            osm_addresses =      pd.concat([osm_addresses, osm_results], sort=False).drop_duplicates()
            osm_addresses.columns=            osm_addresses.columns.set_names(["L0","L1"])

            vlog("osm addresses: ")
            vlog(osm_addresses)

            if rejected.shape[0]>0:
                rejected_addresses = pd.concat([rejected_addresses, rejected], sort=False).drop_duplicates()
                rejected_addresses.columns=     rejected_addresses.columns.set_names(["L0","L1"])


        except Exception as exc:
            osm_results = chunk[[addr_key_field]].copy()
            osm_results[("work", "method")] = "error on " + ";".join(transformers)
            osm_addresses =  pd.concat([osm_addresses, osm_results], sort=False).drop_duplicates()

            log(f"Error during processing: {exc}")
            vlog(traceback.format_exc())

        chunk  = chunk[~chunk[addr_key_field].isin(osm_results[addr_key_field])].copy()
        if chunk.shape[0]==0:
            break

        vlog(step_stats)

    # At this point, ('input', 'streetName'),... contain transformed address. We put back original address

    osm_addresses = osm_addresses.drop([street_field, housenbr_field, postcode_field, city_field, country_field], axis=1, errors="ignore").merge(to_process_addresses, how="left")


    if with_extra_house_number and osm_addresses.shape[0] > 0:
        osm_addresses = add_extra_house_number(osm_addresses)
        
    rejected_addresses = add_dist_to_match(osm_addresses, rejected_addresses)

    return osm_addresses, rejected_addresses #{"match": format_res(osm_results), "rejected": format_res(all_reject)}
