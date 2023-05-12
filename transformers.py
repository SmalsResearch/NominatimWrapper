# -*- coding: utf-8 -*-
"""
Transformers functions

@author: Vandy Berten (vandy.berten@smals.be)
"""

# pylint: disable=line-too-long

from datetime import datetime

import pandas as pd
import numpy as np


from config import (addr_key_field, street_field,
                    housenbr_field, postcode_field,
                    city_field, country_field,
                    regex_replacements,
                    transformed_address_field)

from base import (vlog, get_photon, update_timestats, parse_address)

from check_result_utils import ignore_mismatch_keep_bests



##################################
## Photon
##################################

photon_street_field   = ("photon","street")
# Sometimes, streetname is put in "name" field (especially for request without
#  house number)
photon_name_field     = ("photon","name")
photon_postcode_field = ("photon","postcode")
photon_city_field     = ("photon","city")
photon_country_field  = ("photon","country")


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
                                  secondary_sort_field = ("photon", "photon_order"))
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
    res[("photon","parsed")] = res[photon_col].apply(lambda j:j["features"] if "features" in j
                                                                            else None)

    res = res.set_index([addr_field])

    ser = res[("photon","parsed")].apply(pd.Series, dtype=object)

    if ser.shape[0] == 0 or ser.shape[1] == 0:
        return pd.DataFrame(columns = [addr_field])

    photon_results = pd.DataFrame(ser.stack())
    photon_results.columns = pd.MultiIndex.from_tuples([photon_col], names=["L0", "L1"])

    photon_results = photon_results.reset_index(level=0)
    # photon_results  = pd.DataFrame(photon_results) # uncomment before running pylint...

    addr_items = []

    for row in photon_results[photon_col].apply(lambda x: x["properties"]):
        for addr_item in row.keys():
            addr_items.append(addr_item)

    addr_items = pd.Series(addr_items).value_counts().iloc[0:30].keys().values

    #prefix="photon_"
    for addr_item in addr_items:
        photon_results[("photon", addr_item)] = photon_results[photon_col].apply(lambda x, ad_it=addr_item: x["properties"][ad_it] if ad_it in x["properties"] else None)

    for fld in [photon_street_field, photon_postcode_field, photon_city_field, photon_country_field]:
        if fld not in photon_results:
            vlog(f"Photon: adding field {fld}")
            photon_results[fld] = ""

    if photon_name_field in photon_results:
        photon_results[photon_street_field] = photon_results[photon_street_field].replace("", pd.NA).fillna(photon_results[photon_name_field])

    photon_results[('photon', 'photon_order')] = photon_results.index

    return photon_results

def process_photon(addr_df, addr_field):
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

    photon_col = ("photon", "out")

    to_process = addr_df[[addr_field]].drop_duplicates()

    vlog(f"Photon: Will process {addr_df.shape[0]} with {to_process.shape[0]} unique values")

    to_process[photon_col] = to_process[addr_field].apply(get_photon)

    photon_results = photon_parse_and_split(to_process, addr_field, photon_col)

    vlog(f"Photon got {photon_results.shape[0]} results for {addr_df.shape[0]} addresses")

    if photon_results.shape[0]>0:
        photon_results = addr_df[[addr_key_field, addr_field]].merge(photon_results)
    # else:
        # photon_results = addr_df[[addr_key_field, addr_field]].copy()
        # photon_results[photon_street_field]=pd.NA
        # photon_results[photon_postcode_field]=pd.NA
        # photon_results[photon_city_field]=pd.NA
        # photon_results[("photon", "photon_order")]=pd.NA

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

    photon_addr[("photon", "full_addr_in")] = photon_addr[street_field].fillna("") +", "\
                            + photon_addr[postcode_field].fillna("") + " " \
                            + photon_addr[city_field].fillna("")+", " \
                            + photon_addr[country_field].fillna("")

    # Send to Photon
    photon_res = process_photon(photon_addr, ("photon", "full_addr_in"))

    if photon_res.shape[0] == 0:
        return photon_res



    if check_results : #and photon_check_results:

        photon_res_sel = photon_keep_relevant_results(photon_res, photon_addr)
    else:
        photon_res_sel = photon_res.merge(addresses[[addr_key_field, street_field,
                                                     housenbr_field, postcode_field,
                                                     city_field, country_field]])
    vlog("photon_res: ")
    vlog(photon_res)

    vlog("photon_res_sel: ")
    vlog(photon_res_sel)

    if photon_res_sel.shape[0] == 0:
        return photon_res_sel

    fields = [(street_field, photon_street_field),
              (housenbr_field, housenbr_field), # We do not consider photon house number
              (city_field, photon_city_field),
              (postcode_field, photon_postcode_field),
              (country_field, photon_country_field)]

    fields_out    = [field_in      for field_in, field_photon in fields if field_photon in  photon_res_sel]
    fields_photon = [field_photon  for field_in, field_photon in fields if field_photon in  photon_res_sel]

    vlog(photon_res_sel)
    update_timestats("'t&p > transformer > photon", start_time)
    res= photon_res_sel[[addr_key_field] + fields_photon].rename(columns= {field_photon[1]: field_in[1] for field_in, field_photon in fields}).rename(columns= {"photon":"input"})[[addr_key_field] + fields_out]

    vlog("Photon transformed:")
    vlog(res)
    return res



###########################
## Libpostal transformer
###########################

lpost_street_field   = ("lpost","road")
lpost_housenbr_field = ("lpost","house_number")
lpost_postcode_field = ("lpost","postcode")
lpost_city_field     = ("lpost","city")
lpost_country_field  = ("lpost","country")

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
    libpost_addr[("lpost", "full_addr_in")] = libpost_addr[street_field] + ", "+ libpost_addr[housenbr_field].fillna("")+", "+                    libpost_addr[postcode_field].fillna("") + " " +libpost_addr[city_field].fillna("") +",  " +                    libpost_addr[country_field].fillna("")

    # Apply libpostal
    libpost_addr[("lpost","out")] = libpost_addr[("lpost","full_addr_in")].apply(parse_address)
    libpost_addr[("lpost", "out")] = libpost_addr[("lpost", "out")].apply(lambda lst: {x: y for (y, x) in lst})

    # Split libpostal results
    for field in [lpost_street_field, lpost_housenbr_field, lpost_postcode_field, lpost_city_field, lpost_country_field]:
        libpost_addr[field] =libpost_addr[("lpost", "out")].apply(lambda rec, fld=field[1]: rec[fld] if fld in rec else np.NAN)

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

        return pd.DataFrame(columns=[transformed_address_field, addr_key_field])#,  libpost_addr


    fields =        [(street_field, lpost_street_field), (housenbr_field, lpost_housenbr_field),
                     (city_field, lpost_city_field), (postcode_field, lpost_postcode_field),
                     (country_field, lpost_country_field) ]
    fields_out    = [field_in      for field_in, field_lpost in fields]
    fields_lpost  = [field_lpost   for field_in, field_lpost in fields]

    update_timestats("'t&p > transformer > libpostal", start_time)

    return libpost_addr[[addr_key_field] + fields_lpost].rename(columns= {field_lpost[1]: field_in[1] for field_in, field_lpost in fields}).rename(columns= {"lpost":"input"})[[addr_key_field] + fields_out]



############################
## Regex transformer
###########################

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
        new_values = regex_addr[field].fillna("").str.replace(match, repl,regex=True)
        new_values_sel = regex_addr[field].fillna("") != new_values

        if new_values_sel.sum()>0:
            vlog(regex_addr[new_values_sel])

            regex_addr[field] = new_values
            vlog("-->")
            vlog(regex_addr[new_values_sel])
        else:
            vlog("None")

    return regex_addr
