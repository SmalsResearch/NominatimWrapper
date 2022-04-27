#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import urllib

import numpy as np

import json

from tqdm.autonotebook import  tqdm

import re

# %matplotlib inline

tqdm.pandas()


import jellyfish#88942
import dask.dataframe as dd

from dask.multiprocessing import get
from dask.diagnostics import ProgressBar

from IPython.display import display

from datetime import datetime, timedelta
import logging
import sys

logging.basicConfig(format='[%(asctime)s]  %(message)s', stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


from config import *


# In[ ]:


# !jupyter nbconvert --to python AddressCleanserUtils.ipynb


# In[ ]:


within_jupyter=False


# In[33]:


def log(arg):
    if (type(arg) == pd.core.frame.DataFrame) or (type(arg) == pd.core.frame.Series):
        log_display(arg)
    else:
        logging.info(arg)

def vlog(arg):
    if (type(arg) == pd.core.frame.DataFrame) or (type(arg) == pd.core.frame.Series):
        vlog_display(arg)
    else:
        logging.debug(arg)

        
def log_display(df):
    if within_jupyter: 
        if logger.getEffectiveLevel() <= logging.INFO:
            display(df)
    else: 
        with pd.option_context("display.max_columns", None, 'display.width', 200):
            log("\n"+str(df))
        
def vlog_display(df):
    if within_jupyter: 
        if logger.getEffectiveLevel() <= logging.DEBUG:
            display(df)
    else: 
        with pd.option_context("display.max_columns", None,  'display.width', 200):
            vlog("\n"+str(df))


# In[ ]:


pbar = ProgressBar(dt=1.0)


# In[ ]:


# Mapping of nominatim results fields on our output fields
collapse_params = {
    "addr_out_street":   ["road", "pedestrian","footway", "cycleway", "path", "address27", "construction", "hamlet", "park"],
    "addr_out_city"  :   ["town", "village", "city_district", "county", "city"],
    "addr_out_number":   ["house_number"],
    "addr_out_country":  ["country"],
    "addr_out_postcode": ["postcode"],
}


# In[ ]:


osm_addr_field = "osm_addr" # name of the field of the address sent to Nominatim

similarity_threshold = 0.5


# In[ ]:


timestats = {"transformer": timedelta(0),
             "osm": timedelta(0),
             "osm_post": timedelta(0),
             "checker": timedelta(0),
             "photon": timedelta(0),
             "libpostal": timedelta(0)}


# # Functions 

# ## Global

# In[ ]:


import unicodedata
def remove_accents(input_str):
    if pd.isnull(input_str):
        return None
    
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


# In[ ]:


def house_number_compare(n1, n2):
    """
    Compare two pd.Series of house numbers
    - if n1 == n2 (and not empty) --> 1
    - else we split n1 and n2 in chunks of numbers : 
        - if first chunk of n1 = second chunk of n2 -->  0.8  (e.g : 10 vs 10-12)
        - else second chunk of n1 = first chunk of n2 --> 0.8 (e.g. 10-12 vs 12)
    - else if only numbers are equal (and not empty) --> 0.5  (e.g., '10a' vs '10 B' vs '10')
    """
    n1 = n1.fillna("").astype(str).str.strip()
    n2 = n2.fillna("").astype(str).str.strip()
    
    res= ((n1==n2) & (n1.str.len()>0)).astype(float)
    
    n1_split = n1.str.split("[^0-9]", expand=True)
    n2_split = n2.str.split("[^0-9]", expand=True)
    
    if n2_split.shape[1]>1:
        res[(res == 0) & ((n1_split[0] == n2_split[1]) & (n2_split[1].str.len()>0))] = 0.8
        
    if n1_split.shape[1]>1:
        res[(res == 0) & ((n1_split[1] == n2_split[0]) & (n1_split[1].str.len()>0))] = 0.8
    
    res[(res == 0) & (n1.str.replace("[^0-9]", "") == n2.str.replace("[^0-9]", "")) & ( n1.str.len()>0) & ( n2.str.len()>0)] = 0.5
    
    return res


# In[ ]:


def postcode_compare(s1, s2):
    """Compare postcodes.

    If the postcodes in both records are identical, the similarity
    is 1. If the first two values agree and the last two don't, then
    the similarity is 0.5. Otherwise, the similarity is 0.
    """
    s1 = s1.fillna("").astype(str).str.replace("^[A-Z]-?", "")
    s2 = s2.fillna("").astype(str).str.replace("^[A-Z]-?", "")
    
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
    return 1-jellyfish.damerau_levenshtein_distance(str1, str2)/max(len(str1), len(str2)) if (len(str1) > 0 or len(str2) > 0  ) else 0.0


# In[ ]:





# In[ ]:


def inclusion_test(s1, s2):
    """ 
    Check that a string s1 is equal to another string s2, except that s2 contains an additional substring
    Example : "Avenue C Berten" vs "Avenue Clovis Berten"

    """
    
    import os
    l_pref = len(os.path.commonprefix([s1, s2]))
    l_suf =  len(os.path.commonprefix([s1[::-1], s2[::-1]]))

    res = 1 if (l_pref>0)  and (l_suf > 0) and (l_pref+l_suf >= min(len(s1), len(s2))) else 0
#     if res == 1:
#         print(s1, s2, res)
    return res



# In[ ]:


# s1="NEU"
# s2 = "NEUCHATEAU"
# import os
# l_pref = len(os.path.commonprefix([s1, s2]))
# l_suf =  len(os.path.commonprefix([s1[::-1], s2[::-1]]))
# l_pref, l_suf


# In[ ]:


def fingerprint(column):
    cleaned_column = column.fillna("")
#     cleaned_column = cleaned_column.str.upper().apply(remove_accents)
    cleaned_column = cleaned_column.str.replace("[^A-Z]", " ")
    cleaned_column = cleaned_column.str.strip()
    cleaned_column = cleaned_column.str.split("[ ]+")
    
    cleaned_column = cleaned_column.apply(lambda x: sorted(list(set(x))))
    cleaned_column = cleaned_column.apply(" ".join)
    return cleaned_column


# In[ ]:





# In[ ]:


# TODO : replacement seulement si dans les 2 inputs en même temps  --> Pour éviter que "Avenue Louise" et "Place Louise" aient une similarité de 100%

street_compare_removes = [r"\([A-Z.]+\)", 
                          # r"[^A-Z ]+",
                          r"\b(AVENUE|RUE|CHAUSSEE|BOULEVARD|PLACE)\b",
                          r"(STRAAT|LAAN|STEENWEG|WEG)\b"
                         ]

dontwatchthis = "DONOTCONSIDERTHISSTRING"
def _street_compare(street1, street2, compare_algo, street_compare_removes):
    
    streets = pd.DataFrame()
    streets["STR1"] = street1
    streets["STR2"] = street2
    
    for i in ["STR1", "STR2"]:
        for scr in street_compare_removes:
            streets[i] = streets[i].str.replace(scr, "")
        
        streets[i] = streets[i].str.strip().str.replace(" [ ]+", " ")
    
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
    
def street_compare(street1, street2, compare_algo = levenshtein_similarity):

    if street1.shape[0] == 0:
        return pd.Series(index=street1.index)
    
    # For Brussels (or bilingual regions), we typically get "Avenue Louise - Louizalaan" for street
    # We also often get (in input) streets like "Bruxelles, Avenue Louise", or "Avenue Louise, 10"
    # Set "dontwatchthis" for values that do not split, but where a column appear because of other values being split
    
    street_split_a = street1.fillna("").str.replace(",", " - ").str.split(" - ", expand=True).fillna(dontwatchthis)
    street_split_b = street2.fillna("").str.replace(",", " - ").str.split(" - ", expand=True).fillna(dontwatchthis)

    #display(pd.concat([street1, street2], axis=1))
    #display(pd.concat([street_split_a, street_split_b], axis=1))
    
    street_distances = pd.DataFrame()
    for ai in range(street_split_a.shape[1]):
        str_a = street_split_a[ai].str.upper().apply(remove_accents).str.replace( r"[^A-Z ]+", " ").str.replace(" [ ]+", " ").str.strip()

        for bi in range(street_split_b.shape[1]):
            str_b = street_split_b[bi].str.upper().apply(remove_accents).str.replace( r"[^A-Z ]+", " ").str.replace(" [ ]+", " ").str.strip()

            street_distances["SIM_street_a{}b{}".format(ai,bi)] =  _street_compare(str_a, str_b, compare_algo=compare_algo, street_compare_removes=street_compare_removes)
            # to calculate (strict) inclusion, we do not remove "street words" (Rue, Avenue ...) 
            street_distances["INC_street_a{}b{}".format(ai,bi)] =  _street_compare(str_a, str_b, compare_algo=inclusion_test, street_compare_removes=[])
            
            fgpta = fingerprint(str_a)
            fgptb = fingerprint(str_b)
            
            street_distances["FING_street_a{}b{}".format(ai,bi)] =  _street_compare(fgpta, fgptb, compare_algo=compare_algo, street_compare_removes=street_compare_removes)
    
    street_distances["SIM_street"] =  street_distances[filter(lambda x: "SIM_street_a" in x 
                                                              or "INC_street_a" in x
                                                              or "FING_street_a" in x, street_distances)].max(axis=1)    
#     display(street_distances[street1.fillna("").str.contains("AUTOSNELWEGEN") ])
    vlog(f"Street compare: {street1.name}, {street2.name}")
    vlog(pd.concat([street_split_a, street_split_b, street_distances], axis=1))
    return street_distances["SIM_street"]


# In[ ]:





# In[ ]:


def city_compare(city1, city2, compare_algo = levenshtein_similarity):
    
    cities = pd.DataFrame()
    cities["CITY1"] = city1
    cities["CITY2"] = city2
    
    for i in ["CITY1", "CITY2"]:
        cities[i] = cities[i].str.upper().apply(remove_accents)
        cities[i] = cities[i].str.strip().str.replace(" [ ]+", " ")
        
    return cities.fillna("").apply(lambda row : compare_algo(row.CITY1, row.CITY2), axis=1)


# In[ ]:


def ignore_mismatch_keep_bests(addr_matches, addr_key_field, 
                               street_fields_a, housenbr_field_a, postcode_field_a, city_field_a, 
                               street_field_b, housenbr_field_b, postcode_field_b, city_field_b, #split_comma = True,
                               similarity_threshold=similarity_threshold, max_res=1, secondary_sort_field = "osm_order"):
    
    if addr_matches.shape[0] == 0:
        return addr_matches, addr_matches
    
    distances = pd.DataFrame(index=addr_matches.index)
    
    street_b = addr_matches[street_field_b]
        
    distances["SIM_street"] = -1
    
    vlog("Will compare streets")
    for street_field_a in street_fields_a :
        # Only compute a new street distance if the computed distance is below the threshold so far
        x = (distances["SIM_street"] < similarity_threshold)  
        
        distances.loc[x, "SIM_street"] =  street_compare(addr_matches[street_field_a][x].fillna(""), street_b[x])
        distances.loc[x, "SIM_street_which"] =  street_field_a # last field that have been compared
    
    wsu = " ; ".join([f"{r}: {c}" for r, c in distances[distances["SIM_street"] >= similarity_threshold]["SIM_street_which"].value_counts().iteritems()])
    vlog(f"Which street used: {wsu}")

    
    w = distances[(distances["SIM_street"] >= similarity_threshold)&(distances["SIM_street_which"]!=street_fields_a[0] )].merge(addr_matches, left_index=True, right_index=True)
    vlog(f"Cases where street ({street_fields_a[0]}) wasn't used to validate results: ")
    
    vlog(w[np.concatenate([[addr_key_field], street_fields_a, [housenbr_field_a, postcode_field_a, city_field_a, 
                               street_field_b, housenbr_field_b, postcode_field_b, city_field_b]])])
    
    distances["SIM_house_nbr"] = house_number_compare(addr_matches[housenbr_field_a].fillna(""), addr_matches[housenbr_field_b].fillna(""))
    
    distances["SIM_zip"] =       postcode_compare(addr_matches[postcode_field_a].fillna(""), addr_matches[postcode_field_b].fillna(""))
    
    distances["SIM_city"] =      city_compare(addr_matches[city_field_a].fillna(""), addr_matches[city_field_b].fillna(""))
    
    elimination_rule = ((distances.SIM_zip < 0.1) & (distances.SIM_city < similarity_threshold)) | \
                        ((distances.SIM_street < similarity_threshold)  )
    
    rejected = addr_matches[elimination_rule].merge(distances, left_index=True, right_index=True).copy()
    
    rejected["reject_reason"] = "mismatch"

    # Remove non acceptable results
    
    result = addr_matches[~elimination_rule].merge(distances, left_index=True, right_index=True).sort_values([addr_key_field, "SIM_street", "SIM_house_nbr", secondary_sort_field], ascending=[True, False, False, True])
    
    # Keep only the first ones
    result_head = result.groupby([addr_key_field]).head(max_res)#.drop("level_2", axis=1)#.set_index([key, addr_key_field])#[init_osm.index.get_level_values(1) == 140266	]

    result_tail = result[~result.index.isin(result_head.index)].copy()
    result_tail["reject_reason"] = "tail" 
    
    return result_head, rejected.append(result_tail)


# In[ ]:


def retry_with_low_place_rank(osm_results, sent_addresses, 
                              street_field, housenbr_field,  postcode_field, city_field, country_field,
                              check_results=True, osm_structured=False):
    vlog("Trying to improve place_rank with place_rank < 30 by cleansed house number ")
    sent_addresses_26 = osm_results[osm_results.place_rank < 30].merge(sent_addresses)#[osm_addresses.place_rank == 26]
    
    vlog(f"    - <30: {sent_addresses_26.shape[0]}")
    sent_addresses_26 = sent_addresses_26[~sent_addresses_26[housenbr_field].fillna("").astype(str).str.match("^[0-9]*$")]
    vlog(f"    - numbers: {sent_addresses_26.shape[0]}")
    sent_addresses_26["housenbr_clean"] = sent_addresses_26[housenbr_field].fillna("").astype(str).str.extract("^([0-9]+)")[0]

    sent_addresses_26["osm_addr_in"] =   sent_addresses_26[street_field  ].fillna("") + ", "+ sent_addresses_26["housenbr_clean"].fillna("") +", " + \
                                         sent_addresses_26[postcode_field].fillna("") + " " +sent_addresses_26[city_field    ].fillna("") +", "+ \
                                         sent_addresses_26[country_field].fillna("")

    vlog(" ; ".join([f"rank {r}: {c}" for r, c in sent_addresses_26.place_rank.value_counts().iteritems()]))
    #print(osm_results_26.place_rank.value_counts())
    osm_results_26, rejected_26 = process_osm(sent_addresses_26, 
                                              osm_addr_field="osm_addr_in", addr_key_field=addr_key_field, 
                                              street_field=street_field,housenbr_field="housenbr_clean",  
                                              postcode_field=postcode_field, city_field=city_field,
                                              country_field=country_field,
                                              check_results=check_results,
                                              osm_structured=osm_structured)
    
    if osm_results_26.shape[0]>0:
        vlog("     - New results with place_rank == 30 after cleansing ({}):".format(" ; ".join([f"rank {r}: {c}" for r, c in osm_results_26.place_rank.value_counts().iteritems()])))
        
        osm_results_26 = osm_results_26[osm_results_26.place_rank == 30]
        osm_results_26["retry_on_26"] = True
        
#         display(osm_results_26)
        
        osm_results = osm_results[~osm_results[addr_key_field].isin(osm_results_26[addr_key_field])].append(osm_results_26, sort=False)
        
    return osm_results


# In[71]:


def find_house_number(street, house_number):
    if house_number != "" and not pd.isnull(house_number):
        return house_number

    lpost = parse_address(street)
    lpost = {x: y for (y, x) in lpost}
    return lpost["house_number"] if "house_number" in lpost else np.NaN

def add_extra_house_number(osm_addresses, addresses, street_field, housenbr_field):
    if "addr_out_number" not in osm_addresses:
        return osm_addresses
        
    result = osm_addresses.merge(addresses)
    result["extra_house_nbr"] = result.apply(lambda row: find_house_number(row[street_field], row[housenbr_field]), axis=1)

    return result[np.concatenate([osm_addresses.keys(), ["extra_house_nbr"]])]

def get_lpost_house_number(street):
    lpost = parse_address(street)
    housenbr = ";".join([y for (y, x) in lpost if x=="house_number"])
    boxnbr = ";".join([y for (y, x) in lpost if x=="unit"])
#     log(f"get_lp : '{street}' - {housenbr} - {boxnbr}")
#     log(lpost)
    #return {"lpost_house_nbr": housenbr, "lpost_box_nbr": boxnbr}
    return [housenbr, boxnbr]
    
def add_extra_house_number(osm_addresses, addresses, street_field, housenbr_field, city_field, postcode_field ):
    vlog("Start adding extra house number")
    if "addr_out_number" not in osm_addresses:
        return osm_addresses
    
    
    result = osm_addresses.merge(addresses)
    result["in_house_nbr"] = result[housenbr_field]
    
    lp = result.fillna("").apply(lambda row: get_lpost_house_number(f"{row[street_field]} {row[housenbr_field]}, {row[postcode_field]} {row[city_field]}".strip()), axis=1,  result_type ='expand')
    
    #lp= (result[street_field] + " " + result[housenbr_field]).apply(get_lpost_house_number, result_type ='expand')#, axis=1)
#     log(f"lp: {lp}")
    result[["lpost_house_nbr", "lpost_unit"]] = lp
    vlog("End of  adding extra house number")
    return result[np.concatenate([osm_addresses.keys(), ["in_house_nbr", "lpost_house_nbr", "lpost_unit"]])]


# In[64]:


# def parse_address(street):
#     return [("20",  "house_number"), ("a", "unit")]
# osm_addresses = pd.DataFrame({"addr_key" : [1], "addr_out_number": [""]})
# addresses = pd.DataFrame({"addr_key" : [1], "street_field": ["avenue fonsnsy 30"], "housenbr_field": [""]})

# add_extra_house_number(osm_addresses, addresses, "street_field", "housenbr_field")


# In[29]:





# In[ ]:


def transform_and_process(to_process_addresses, transformers, addr_key_field, street_field, housenbr_field, 
                          city_field, postcode_field, country_field, check_results=True, osm_structured=False):

    t = datetime.now()
    method = "+".join(transformers)
    
    # second test : to get rid of "phantom dask partition"
    if to_process_addresses.shape[0]==0 or to_process_addresses[addr_key_field].duplicated().sum() > 0:
        
        vlog("No more addresses!")
        step_stats = {"method": method, "todo":  0, "sent": 0, "match": 0, "match_26": 0, 
                      "reject_rec" :0, "reject_addr": 0, "reject_mism": 0}
        return pd.DataFrame(columns=[addr_key_field]), pd.DataFrame(columns=[addr_key_field, "reject_reason"]), step_stats

    
    transformed_addresses = apply_transformers(to_process_addresses, transformers, addr_key_field, 
                                               street_field=street_field, housenbr_field=housenbr_field, 
                                               postcode_field=postcode_field, city_field=city_field, country_field=country_field,
                                               check_results=check_results)
    

    if transformed_addresses.shape[0]==0:
        vlog("No more addresses for this transformers sequence!")
        step_stats = {"method": method, "todo":  0, "sent": 0, "match": 0, "match_26": 0, "reject_rec" :0, "reject_addr": 0, "reject_mism": 0}
        return pd.DataFrame(columns=[addr_key_field]), pd.DataFrame(columns=[addr_key_field, "reject_reason"]), step_stats

    transformed_addresses["osm_addr_in"] =   transformed_addresses[street_field  ].fillna("") + ", "+ \
                                             transformed_addresses[housenbr_field].fillna("") + ", "+ \
                                             transformed_addresses[postcode_field].fillna("") + " " +\
                                             transformed_addresses[city_field    ].fillna("") + ", "+\
                                             transformed_addresses[country_field    ].fillna("") 
    
    
    transformed_addresses["osm_addr_in"]= transformed_addresses["osm_addr_in"].str.replace("^[ ,]+", "")

    if check_with_transformed :
        sent_addresses = transformed_addresses
    else:
        sent_addresses = transformed_addresses[["osm_addr_in", addr_key_field]].merge(to_process_addresses, on=addr_key_field)
    
    vlog(f"Will process {sent_addresses.shape[0]} addresses for : transformers = {'+'.join(transformers)}")

    vlog(sent_addresses.head())
    vlog(sent_addresses.shape)

    timestats["transformer"] += datetime.now() - t
         
    osm_results, rejected = process_osm(sent_addresses, 
                                        osm_addr_field="osm_addr_in", addr_key_field=addr_key_field, 
                                        street_field=street_field, housenbr_field=housenbr_field, 
                                        postcode_field=postcode_field, city_field=city_field,
                                        country_field=country_field,
                                        check_results=check_results,
                                        osm_structured=osm_structured)
    
    if with_cleansed_number_on_26 and osm_results.shape[0]>0 : 

        osm_results = retry_with_low_place_rank(osm_results, sent_addresses, 
                                                street_field=street_field,housenbr_field=housenbr_field,  
                                                postcode_field=postcode_field, city_field=city_field,
                                                country_field=country_field,
                                                check_results=check_results)

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

# In[14]:


def get_osm(addr, accept_language = ""): #lg = "en,fr,nl"
    params = urllib.parse.urlencode({"q": addr,
                                    "format":"jsonv2",
                                    "accept-language":accept_language,
                                    "addressdetails":"1",
                                    "namedetails" : "1",
                                    "limit": "50"
                                    })
    
    url = "http://%s/search.php?%s"%(osm_host, params)
    vlog(f"Call to OSM: {url}")
    try: 
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             return res
            return [ {field: item[field] for field in ["place_id", "lat", "lon", "display_name", "address", "namedetails", "place_rank", "category", "type"]} for item in res] 
    except Exception as e:
        raise Exception (f"Cannot get OSM results ({osm_host}): {e}") 


# In[2]:


def get_osm_struct(street, housenumber, postcode, city, country, accept_language = ""): #lg = "en,fr,nl"
    params = urllib.parse.urlencode({"street": f"{street}, {housenumber}" if pd.notnull(street) and len(str(street).strip())>0 else "" ,
                                     "city":city,
                                     "postalcode": postcode,
                                     "country": country,
                                    "format":"jsonv2",
                                    "accept-language":accept_language,
                                    "addressdetails":"1",
                                    "namedetails" : "1",
                                    "limit": "50"
                                    })
    
    url = "http://%s/search.php?%s"%(osm_host, params)
    vlog(f"Call to OSM: {url}")
    try: 
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             return res
            return [ {field: item[field] for field in ["place_id", "lat", "lon", "display_name", "address", "namedetails", "place_rank", "category", "type"]} for item in res] 
    except Exception as e:
        raise Exception (f"Cannot get OSM results ({osm_host}): {e}") 


# In[24]:


# osm_host="10.1.0.45:8081"
# get_osm_struct(city="Auderghem", street=None, housenumber=None, postcode="1160", country="Belgique")


# In[21]:


# osm_host="10.0.2.15:7070"
# get_osm_struct("avenue fonsny", "20", "1060", "bruxelles", "Belgique" )


# In[23]:


def get_osm_details(place_id): #lg = "en,fr,nl"
    params = urllib.parse.urlencode({"place_id": place_id,
                                    "format":"json",
                                    })

    url = "http://%s/details.php?%s"%(osm_host, params)
    try: 
        with urllib.request.urlopen(url) as response:
            res = response.read()
            return json.loads(res)
    except Exception as e: 
        logger.warning(f"Problems with get_details for place_id {place_id} (url: {url})")
        print(e)
        return {"category":"error", "names": []}
#         raise e
    


# In[ ]:





# In[16]:


def process_osm(df, osm_addr_field, addr_key_field, street_field, housenbr_field, 
                postcode_field, city_field, country_field, accept_language="", similarity_threshold=similarity_threshold, 
               check_results=True, osm_structured=False) :
    
    t = datetime.now()
    
    if df.shape[0] == 0:
        
        return pd.DataFrame(columns=[osm_addr_field, addr_key_field]), pd.DataFrame(columns=[osm_addr_field, addr_key_field, "reject_reason"])
    
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
     
#     if with_dask : 
#         dd_to_process = dd.from_pandas(to_process, chunksize=1000 ) #, npartitions=10)

#         dask_task = dd_to_process[[osm_addr_field, "accept_language"]].apply(lambda row: get_osm(row[osm_addr_field], row["accept_language"]), meta=('x', 'str'), axis=1)

#         to_process[osm_res_field] = dask_task.compute() #dchunk.loc[:].apply(map_partitions(lambda r: get_osm(r[addr_field])).compute()#get=get)
#     else: 
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
        
    timestats["osm"] += datetime.now() - t
    
    t = datetime.now()
#     to_process.to_pickle("osm_raw.pkl")
    
#     to_process = pd.read_pickle("osm_raw.pkl")

    vlog("     - Parse & split osm results ...")

    osm_results = osm_parse_and_split(to_process, osm_res_field, osm_addr_field=osm_addr_field)
    
    to_process = None # Allows Garbage collector to free memory ...

    osm_results = df[[osm_addr_field, addr_key_field]].merge(osm_results)
    
    vlog(f"     - OSM got {osm_results.shape[0]} results for {osm_results[addr_key_field].nunique()} addresses")
    
    timestats["osm_post"] += datetime.now() - t
#     display(osm_results)
    
#     osm_results.to_pickle("osm_parsed.pkl")
    
#     display(osm_results)
    if osm_results.shape[0] == 0:
        
        return osm_results, pd.DataFrame(columns=[osm_addr_field, addr_key_field, "reject_reason"])

    if check_results:
        
        t = datetime.now()
    
        vlog("     - Keep relevant results")
        osm_results, osm_reject = osm_keep_relevant_results(osm_results, df, street_field, housenbr_field, 
                                                            postcode_field, city_field, country_field, 
                                                            similarity_threshold=similarity_threshold, addr_key_field=addr_key_field)


        vlog(f"     - Got {osm_results.shape[0]} results")

        if use_osm_parent: 
            vlog("     - Trying alternative (parent) names for rejected answers")

            # Keep rejected records that do not correspond to an accepted address
            final_rejected = osm_reject[(osm_reject.reject_reason == "mismatch") & 
                                        (~osm_reject[addr_key_field].isin(osm_results[addr_key_field]))]

            # Get parent place id from place id calling get_osm_details
            parent_place_id = final_rejected.place_id.apply(get_osm_details).apply(lambda x: (x["parent_place_id"] if "parent_place_id" in x else 0 ))
            
            if (parent_place_id == 0).any():
                log("Got some parent_place_id == 0")
                log(final_rejected[parent_place_id == 0])
                parent_place_id = parent_place_id[parent_place_id != 0]

            # Get alt names from details of parent
            alt_names =  parent_place_id.apply(get_osm_details).apply(lambda x: (x["names"], x["category"])).apply(pd.Series)

            # Keep only street parents, and split columns ("name", "name:"fr", "old_name" ...) in rows

            if alt_names.shape[0] >0 and alt_names[alt_names[1] == "highway"].shape[0] >0 :
                alt_names = alt_names[alt_names[1] == "highway"]
                alt_names = alt_names[0].apply(pd.Series).stack().reset_index(1).rename(columns= {0: "alt_names"})
                alt_names = final_rejected.merge(alt_names, left_index=True, right_index=True)

                # Keep only alt names that are different from street name
                alt_names = alt_names[alt_names.addr_out_street != alt_names.alt_names]

                # Remove "old" similarity values
                alt_names = alt_names.drop([f for f in alt_names if "SIM" in f], axis=1)

                keep, reject  = ignore_mismatch_keep_bests(alt_names, addr_key_field, 
                                              street_fields_a = ["alt_names"],   housenbr_field_a = "addr_out_number", postcode_field_a = "addr_out_postcode", city_field_a = "addr_out_city",
                                              street_field_b = street_field,  housenbr_field_b = housenbr_field,    postcode_field_b = postcode_field,      city_field_b = city_field,)


                osm_results = osm_results.append(keep, sort=False)
            #     print(osm_reject.shape)
                osm_reject = osm_reject[~ osm_reject[[addr_key_field,"place_id"]].astype(str).apply(";".join, axis=1).isin(keep[[addr_key_field, "place_id"]].astype(str).apply(";".join, axis=1)) ] 
            #     print(osm_reject.shape)
                vlog("     - Saved : ")
                vlog(keep)
            else:
                vlog("     - Not any alt name")
        
        timestats["checker"] += datetime.now() - t
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
    


# In[25]:


def osm_parse_and_split(df, osm_res_field, osm_addr_field, prefix="addr_", drop_osm=True):
    osm_result_item_field = "osm_item_result"

    df = df.set_index([osm_addr_field])
    
    vlog("        * Unstacking...")
    df_split = df[osm_res_field].apply(pd.Series)
    
    if df_split.shape[1] == 0: # None of the addresses were matched by Nominatim
        return pd.DataFrame(columns = [osm_addr_field])
    
    osm_results = pd.DataFrame(df_split.stack()).rename(columns = {0:osm_result_item_field})
    
    vlog("        * Extract items")

    for item in osm_results.iloc[0][osm_result_item_field].keys() :
        osm_results[item] = osm_results[osm_result_item_field].apply(lambda x: x[item] if item in x else None)
    
    
    addr_items = []

    for row in osm_results[osm_result_item_field].apply(lambda x: x["address"]):
        for addr_item in row.keys():
            addr_items.append(addr_item)
            
    addr_items = pd.Series(addr_items).value_counts().keys().values
    
    for addr_item in addr_items:
        osm_results[prefix+addr_item] = osm_results[osm_result_item_field].apply(lambda x: x["address"][addr_item] if addr_item in x["address"] else None)
        
    # Keep only "namedetails" if category == "highway"
    osm_results["namedetails"] = np.where(osm_results["category"] == "highway", osm_results["namedetails"].apply(lambda dct: " - ".join(dct.values())), "")
    
    osm_results = osm_results.drop(columns=["address"] + ([osm_result_item_field] if drop_osm else []))
    
    osm_results.place_rank=osm_results.place_rank.astype(int)
    
    osm_results = add_addr_out_columns(osm_results, prefix)
    
    osm_results = osm_results.reset_index().rename(columns={"level_1": "osm_order"})
    
    return osm_results


# In[26]:


def osm_keep_relevant_results(osm_results, addresses, street_field,  housenbr_field, postcode_field, city_field, country_field, similarity_threshold,
                              addr_key_field, max_res=1):
    
    osm_results_street = osm_results.reset_index().merge(addresses[[addr_key_field, street_field, postcode_field, housenbr_field, city_field, country_field]], left_on=addr_key_field, right_on=addr_key_field, how="left").set_index(osm_results.index)
    
    assert osm_results_street.shape[0] == osm_results.shape[0]

    keep, reject  = ignore_mismatch_keep_bests(osm_results_street, addr_key_field, 
                                      street_fields_a = ["addr_out_street", "addr_out_other", "namedetails"], housenbr_field_a = "addr_out_number", postcode_field_a = "addr_out_postcode", city_field_a = "addr_out_city", 
                                      street_field_b = street_field,      housenbr_field_b = housenbr_field,    postcode_field_b = postcode_field,      city_field_b = city_field,
                                      secondary_sort_field = "osm_order", 
                                      max_res=max_res)
    
    return keep, reject


# In[27]:


def collapse(df, columns, prefix=None, method="fillna"):
    
    if prefix: 
        columns = [prefix+col for col in columns]
        
    if method=="fillna":
        
        res = pd.Series(index = df.index)#[columns[0]]

        for col in columns:
            if col in df.keys():
                res = res.fillna(df[col])
    elif method== "set":
        res = df[columns].apply(lambda lst: set([x for x in lst if not pd.isnull(x)]), axis=1).apply(" - ".join)
    
    else :
        raise Exception("Wrong method ! " + method)
        
    return res


# In[28]:


def add_addr_out_columns(osm_results, prefix):
    
    other_columns = osm_results.keys()

    for out_column in collapse_params:
        other_columns = [col for col in other_columns if col.replace(prefix, "") not in collapse_params[out_column] and col.startswith(prefix)]
    other_columns.remove('addr_country_code')
    
    if "addr_state" in other_columns:
        other_columns.remove('addr_state')
    
    for out_column in collapse_params:
        osm_results[out_column] = collapse(osm_results, collapse_params[out_column], "addr_", "fillna")
        
    osm_results["addr_out_other"] = collapse(osm_results, other_columns, "", "set") if len(other_columns)>0 else np.NaN
    
    return osm_results


# # Transformers

# In[1]:


def apply_transformers(addresses, transformers, addr_key_field, street_field, housenbr_field, postcode_field, 
                       city_field, country_field, check_results):
    
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
            gr = re.match(r"regex\[([a-z]+)\]", transformer)
            regex_key = gr.groups(0)[0]
            
            transformed_addresses =  regex_transformer(transformed_addresses, addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field)
            
        elif transformer == "nonum":
            #transformed_addresses = transformed_addresses[transformed_addresses[housenbr_field].fillna("").str.len()>0].copy()
            transformed_addresses[housenbr_field] = ""

        elif transformer == "nostreet":
            #transformed_addresses = transformed_addresses[transformed_addresses[housenbr_field].fillna("").str.len()>0].copy()
            transformed_addresses[housenbr_field] = ""
            transformed_addresses[street_field] = ""
        
        elif transformer == "nozip":
            transformed_addresses[postcode_field] = ""

        elif transformer == "nocountry":
            transformed_addresses[country_field] = ""

        elif transformer == "libpostal": 
            transformed_addresses = libpostal_transformer(transformed_addresses, addr_key_field, street_field, housenbr_field, postcode_field, 
                                                          city_field, country_field, check_results)
#             display(transformed_addresses)
            
        elif transformer == "photon": 
            transformed_addresses = photon_transformer(transformed_addresses, addr_key_field, street_field, housenbr_field, postcode_field, 
                                                       city_field, country_field, check_results)
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

# In[30]:


photon_street_field   = "photon_street"
photon_name_field     = "photon_name" # Sometimes, streetname is put in "name" field (especially for request without house number)
photon_postcode_field = "photon_postcode"
photon_city_field     = "photon_city"
photon_country_field  = "photon_country"


# In[31]:


def get_photon(addr):
    params = urllib.parse.urlencode({"q": addr})
    url = "http://%s/api?%s"%(photon_host, params)

    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            return json.loads(res)
    except Exception as e:
         raise Exception (f"Cannot connect to Photon ({photon_host}):  {e}")
    


# In[32]:


def photon_keep_relevant_results(photon_results, addresses, 
                                 addr_street_field, addr_housenbr_field, addr_postcode_field, addr_city_field, addr_country_field,
                                 addr_key_field, similarity_threshold):
    
    photon_ext = photon_results.merge(addresses[[addr_key_field,  addr_street_field, addr_housenbr_field, addr_postcode_field, 
                                                 addr_city_field, addr_country_field]])
    
    if photon_ext.shape[0] == 0:
        return pd.DataFrame()
    
    photon_ext["fake_house_number"] = ""
    
#     display(photon_ext)
    vlog("Will compare photon results: ")
    vlog(photon_ext)
    keep, reject  = ignore_mismatch_keep_bests(photon_ext, addr_key_field, 
                                  street_fields_a = [photon_street_field], housenbr_field_a = "fake_house_number", postcode_field_a = photon_postcode_field, city_field_a = photon_city_field, 
                                  street_field_b =   addr_street_field, housenbr_field_b = "fake_house_number", postcode_field_b =   addr_postcode_field, city_field_b =  addr_city_field,
                                     secondary_sort_field = "photon_order")
    
    return keep


# In[33]:


def photon_parse_and_split(res, addr_field, photon_col):
    res["photon_parsed"] = res[photon_col].apply(lambda j:j["features"] if "features" in j else None)
    
    res = res.set_index([addr_field])
    
    s = res.photon_parsed.apply(pd.Series)
#     print(s.shape)
    if s.shape[0] == 0 or s.shape[1] == 0:
        return pd.DataFrame(columns = [addr_field])
    
    photon_results = pd.DataFrame(s.stack()).rename(columns = {0:photon_col})
    
    for item in photon_results[photon_col].apply(lambda x: x.keys())[0]:
        photon_results[item] = photon_results[photon_col].apply(lambda x: x[item] if item in x else None)
   
    addr_items = []

    for row in photon_results[photon_col].apply(lambda x: x["properties"]):
        for addr_item in row.keys():
            addr_items.append(addr_item)

    addr_items = pd.Series(addr_items).value_counts().iloc[0:30].keys().values

    prefix="photon_"
    for addr_item in addr_items:
        #print(addr_item)
        photon_results[prefix+addr_item] = photon_results[photon_col].apply(lambda x: x["properties"][addr_item] if addr_item in x["properties"] else None)
    
    for f in [photon_street_field, photon_postcode_field, photon_city_field, photon_country_field]:
        if f not in photon_results:
            vlog(f"Photon: adding field {f}")
            photon_results[f] = ""
    
    if photon_name_field in photon_results:
        photon_results[photon_street_field] = photon_results[photon_street_field].replace("", pd.NA).fillna(photon_results[photon_name_field])
    
    photon_results["lat"] = photon_results["geometry"].apply(lambda x: x["coordinates"][0])
    photon_results["lon"] = photon_results["geometry"].apply(lambda x: x["coordinates"][1])

    photon_results = photon_results.drop([photon_col, "geometry", "photon_extent", "type","properties", "photon_osm_id"], axis=1, errors="ignore").reset_index().rename(columns={"level_1": "photon_order"})#res #parse_and_split(res, osm_field, key=addr_field)
    return photon_results


# In[34]:


def process_photon(df, addr_field, photon_col, addr_key_field):
    to_process = df[[addr_field]].drop_duplicates()
    
    vlog(f"Photon: Will process {df.shape[0]} with {to_process.shape[0]} unique values")
    
#     if with_dask : 
#         dd_to_process = dd.from_pandas(to_process, npartitions=10)

#         dask_task = dd_to_process[addr_field].apply(get_photon, meta=('x', 'str'))

#         to_process[photon_col] = dask_task.compute()
#     else: 
    to_process[photon_col] = to_process[addr_field].apply(get_photon)
        
    photon_results = photon_parse_and_split(to_process, addr_field, photon_col)
    
    vlog(f"Photon got {photon_results.shape[0]} results for {df.shape[0]} addresses")
    vlog(photon_results)
    
    photon_results = df[[addr_key_field, addr_field]].merge(photon_results)
    
    return photon_results
 


# In[35]:


def photon_transformer(addresses, addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field,
                       check_results, similarity_threshold=similarity_threshold):
    
    t = datetime.now() 
    photon_addr = addresses[[addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field]].copy()
    
    photon_addr["photon_full_addr"] = photon_addr[street_field].fillna("") +", "+ \
                                photon_addr[postcode_field].fillna("") + " " +photon_addr[city_field].fillna("")+", "+ \
                                photon_addr[country_field].fillna("") 
    
    # Send to Photon
    photon_res = process_photon(photon_addr, "photon_full_addr", "photon", addr_key_field = addr_key_field)

    if photon_check_results:

        photon_res_sel = photon_keep_relevant_results(photon_res, photon_addr, addr_street_field=street_field, 
                                                        addr_housenbr_field = housenbr_field,
                                                        addr_postcode_field = postcode_field,  addr_city_field = city_field,
                                                        addr_country_field  = country_field,
                                                        addr_key_field = addr_key_field,  similarity_threshold=similarity_threshold)
    else:
         photon_res_sel = photon_res.merge(addresses[[addr_key_field,  street_field, housenbr_field, postcode_field, 
                                                 city_field, country_field]])
            
    if photon_res_sel.shape[0] == 0:
        return photon_res_sel
    
    fields = [(street_field, photon_street_field), (housenbr_field, housenbr_field), # We do not consider photon house number
              (city_field, photon_city_field), (postcode_field, photon_postcode_field),
              (country_field, photon_country_field)]
    
    fields_out    = [field_in      for field_in, field_photon in fields]
    fields_photon = [field_photon  for field_in, field_photon in fields]
   
    timestats["photon"] += datetime.now() - t
    return photon_res_sel[[addr_key_field] + fields_photon].rename(columns= {field_photon: field_in for field_in, field_photon in fields})[[addr_key_field] + fields_out]


# ## Libpostal

# In[25]:


if with_rest_libpostal:
    import urllib
    # Assuming LibpostalREST flask is running
    def parse_address(address):
        
        import requests

        url = "http://%s/parser"%(libpostal_host)
        params = {"query": address}

        try: 
            res = requests.post(url, json = params)
        except Exception as e:
             raise Exception (f"Cannot connect to Libpostal ({libpostal_host}): {e}")
    
        res = json.loads(res.content.decode())

        return res
    
else: 
    from postal.parser import parse_address


# In[37]:


lpost_street_field   = "lpost_road"
lpost_housenbr_field = "lpost_house_number"
lpost_postcode_field = "lpost_postcode"
lpost_city_field     = "lpost_city"
lpost_country_field  = "lpost_country"



# In[38]:


def libpostal_transformer(addresses, addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field,
                          check_results, similarity_threshold = similarity_threshold):
    
    t = datetime.now() 
    
    libpost_addr = addresses[[addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field]].copy()

    # Make full address for libpostal
    
    libpost_addr["lpost_full_addr_in"] = libpost_addr[street_field] + " "+ libpost_addr[housenbr_field].fillna("")+", "+\
                    libpost_addr[postcode_field].fillna("") + " " +libpost_addr[city_field].fillna("") +",  " +\
                    libpost_addr[country_field].fillna("")
    
    # Apply libpostal
    
    libpost_addr["lpost"] = libpost_addr.lpost_full_addr_in.apply(parse_address)
    libpost_addr["lpost"] = libpost_addr.lpost.apply(lambda lst: {x: y for (y, x) in lst})
    
    # Split libpostal results
    for field in "road", "house_number", "postcode", "city", "house", "country":
        libpost_addr["lpost_"+field] =libpost_addr.lpost.apply(lambda rec: rec[field] if field in rec else np.NAN)
            
    if check_results:
        # Keep only "close" results
        libpost_addr, reject  = ignore_mismatch_keep_bests(libpost_addr, addr_key_field, 
                                      street_fields_a = [street_field], housenbr_field_a = housenbr_field, postcode_field_a = postcode_field,       city_field_a = city_field,  
                                      street_field_b = lpost_street_field,   housenbr_field_b = lpost_housenbr_field, postcode_field_b = lpost_postcode_field, city_field_b = lpost_city_field, 
                                                  secondary_sort_field=addr_key_field)
        vlog("Rejected lipbostal results: ")
        vlog(reject)
    if libpost_addr.shape[0] == 0:
        
        return pd.DataFrame(columns=[osm_addr_field, addr_key_field])#,  libpost_addr
        
    
    fields =        [(street_field, lpost_street_field), (housenbr_field, lpost_housenbr_field), 
                     (city_field, lpost_city_field), (postcode_field, lpost_postcode_field),
                     (country_field, lpost_country_field) ]
    fields_out    = [field_in      for field_in, field_lpost in fields]
    fields_lpost  = [field_lpost   for field_in, field_lpost in fields]
   
    timestats["libpostal"] += datetime.now() - t
    return libpost_addr[[addr_key_field] + fields_lpost].rename(columns= {field_lpost: field_in for field_in, field_lpost in fields})[[addr_key_field] + fields_out]



# ## Regex transformer

# In[39]:


def regex_transformer(addresses, addr_key_field, street_field, housenbr_field, postcode_field, city_field, 
                      country_field, regex_key="init"):
    
    regex_addr = addresses[[addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field]].copy()

    for (field, match, repl) in regex_replacements[regex_key]:
        vlog(f"{field}: {match}")
#         display(regex_addr[field])
        new_values = regex_addr[field].fillna("").str.replace(match, repl)
        new_values_sel = regex_addr[field].fillna("") != new_values
        
        if new_values_sel.sum()>0:
            vlog(regex_addr[new_values_sel])

            regex_addr[field] = new_values
            vlog("-->")
            #display(libpost_addr[libpost_addr[field].fillna("").str.contains(match)])
            vlog(regex_addr[new_values_sel])
        else: 
            vlog("None")

    return regex_addr


