#!/usr/bin/env python
# coding: utf-8

# In[10]:


from flask import Flask,  request,jsonify

import pandas as pd


import os

from  importlib import reload
import AddressCleanserUtils
reload(AddressCleanserUtils)
from AddressCleanserUtils import *

from IPython.display import display

import json

import sys, traceback

from datetime import datetime, timedelta
import time


# In[ ]:


import config_REST
reload(config_REST)
from config_REST import *


# In[ ]:


# TO RUN : 
# jupyter nbconvert --to python AddressCleanserREST.ipynb
# export  FLASK_APP=AddressCleanserREST.py ; export  FLASK_ENV=development ;  flask   run  

# OR : 
# gunicorn -w 8 -b 127.0.0.1:5000 wsgi:app


# In[ ]:


# !jupyter nbconvert --to python AddressCleanserREST.ipynb


# In[ ]:


# AddressCleanserUtils.pbar.unregister()

AddressCleanserUtils.with_dask         = False
AddressCleanserUtils.check_results = True

AddressCleanserUtils.addr_key_field  = addr_key_field

AddressCleanserUtils.regex_replacements = regex_replacements

AddressCleanserUtils.use_osm_parent = use_osm_parent 


# In[ ]:


import logging, os

logger = logging.getLogger()

# WARNING : no logs
# INFO : a few logs
# DEBUG : lots of logs


env_log_level = os.getenv('LOG_LEVEL', "MEDIUM").upper().strip()
if env_log_level == "LOW":
    logger.setLevel(logging.WARNING)
elif env_log_level == "MEDIUM":
    logger.setLevel(logging.INFO)
elif env_log_level == "HIGH":
    logger.setLevel(logging.DEBUG)
else :
    print(f"Unkown log level '{env_log_level}'. Should be LOW/MEDIUM/HIGH")



# In[ ]:


with_timing = os.getenv('TIMING', "NO").upper().strip()
if with_timing == "NO":
    with_timing_info = False
elif with_timing == "YES":
    with_timing_info = True
else: 
    print(f"Unkown TIMING '{with_timing}'. Should be YES/NO")
    with_timing_info = False
log(f"TIMING: {with_timing_info} ({with_timing})")


# In[ ]:


def get_init_df(data):
    return pd.DataFrame([{addr_key_field : "1",
                          street_field:   data["street"],
                          housenbr_field: data["housenumber"],
                          postcode_field: data["postcode"],
                          city_field:     data["city"],
                          country_field:  data["country"]
                          }])


# In[ ]:


def get_row_dict(row, orig=False):
    if orig: 
        return row["osm_item_result"]
    else: 
        to_copy_field = ["osm_id", "place_id", "lat","lon","display_name", "place_rank", "method", 
                         "extra_house_nbr", "in_house_nbr", "lpost_house_nbr", "lpost_unit", "reject_reason", "osm_addr_in"] + list(collapse_params.keys())  + list(filter(lambda x: x.startswith("SIM"), row.index))
        res =  {}

        for f in to_copy_field:
            if f in row: 
                res[f] = row[f]

        return res


# In[ ]:


def format_res(res):
    return list(res.fillna("").apply(lambda row: get_row_dict(row, False), axis=1))


# In[ ]:


default_transformers_sequence = [ ["orig"],
                          ["regex[init]"],
                          ["nonum"],
                          ["libpostal", "regex[lpost]"], 
                          ["libpostal", "regex[lpost]", "nonum"], 
                          ["libpostal", "regex[lpost]", "photon"], 
                          ["libpostal", "regex[lpost]", "photon", "nonum"], 
                          ["photon"],
                          ["photon", "nonum"],
                          ["nostreet"]
                        ]


transformers_sequence = os.getenv('TRANSFORMERS', default_transformers_sequence)

if isinstance(transformers_sequence, str):
    try:
        transformers_sequence = json.loads(transformers_sequence)
    except json.decoder.JSONDecodeError as er:
        log("Cannot parse TRANSFORMERS parameter... uses the default one")
        transformers_sequence = default_transformers_sequence

    
    
vlog("Transformers:")
vlog(transformers_sequence)


# In[1]:


city_test_from = "Bruxelles"
city_test_to = ["Bruxelles", "Brussels"]


# In[2]:





# In[ ]:


# Check that Nominatim server is running properly
# Adapt with the city of your choice!
delay=5
for i in range(10):
    osm = None
    try: 
        osm = get_osm(city_test_from)
        assert city_test_to[0] in osm[0]["namedetails"]["name:fr"] 

        log("Nominatim working properly")
        break

    except Exception as e: 
        log("Nominatim not up & running")
        log(f"Try again in {delay} seconds")
        if osm is not None:
            log("Answer:")
            log(osm)
        
        log(f"Nominatim host: {AddressCleanserUtils.osm_host}")
        
        #raise e
        time.sleep(delay)
if i == 9:
    log("Nominatim not up & running !")
    log(f"Nominatim: {AddressCleanserUtils.osm_host}")


# In[ ]:


# Check that Libpostal is running properly
delay=5
for i in range(10):
    lpost=None
    try: 
        lpost = parse_address(city_test_from)
        assert lpost[0][0].lower() == city_test_to[0].lower()
        log("Libpostal working properly")
        break
        
    except Exception as e: 
        log("Libpostal not up & running ")
        log(f"Try again in {delay} seconds")
        if lpost is not None:
            log("Answer:")
            log(lpost)
            
        time.sleep(delay)
    #raise e
if i == 9:
    log("Libpostal not up & running !")
    log(f"Libpostal: {AddressCleanserUtils.libpostal_host}")
    


# In[ ]:


# Check that Photon server is running properly
delay=5
for i in range(10):
    try:
        ph=""
        ph = get_photon("Bruxelles")
        assert city_test_to[0] in ph["features"][0]["properties"]["name"] or city_test_to[1] in ph["features"][0]["properties"]["name"] 
        log("Photon working properly")
        break


    except Exception as e: 
        log("Photon not up & running ")
        log(f"Try again in {delay} seconds")
        log(ph)
        time.sleep(delay)
        
        #raise e
if i == 9:
        log("Photon not up & running ! Start it with 'nohup java -jar photon-*.jar &'")
        log(f"Photon host: {AddressCleanserUtils.photon_host}")


# In[ ]:





# In[ ]:


def process_address(data, check_results=True, osm_structured=False, with_extra_house_number=True):
    vlog(f"Will process {data}")
    t = datetime.now()
    to_process_addresses = get_init_df(data)
    #update_timestats("init_df", t)
    
    vlog("Got dataframe")
    all_reject = pd.DataFrame()
    for transformers in transformers_sequence:
        vlog ("--------------------------")
        vlog("| Transformers : " + ";".join(transformers))
        vlog ("--------------------------")

        try :
            t = datetime.now()
            osm_results, rejected, step_stats = transform_and_process(to_process_addresses, transformers, addr_key_field, 
                                                                      street_field=street_field, housenbr_field=housenbr_field, 
                                                                      postcode_field=postcode_field, city_field=city_field,
                                                                      country_field=country_field,
                                                                      check_results=check_results,
                                                                      osm_structured=osm_structured)
            update_timestats("t&p", t)
        except Exception as e: 
            log(f"Error during processing : {e}")
            traceback.print_exc(file=sys.stdout)
            return {"error": str(e)}
        
        all_reject = all_reject.append(rejected, sort=False)
        
        vlog(step_stats)
        if osm_results.shape[0] > 0:
            if with_extra_house_number :
                osm_results = add_extra_house_number(osm_results, to_process_addresses, 
                                                     street_field=street_field, housenbr_field=housenbr_field,
                                                     postcode_field=postcode_field, city_field=city_field)
            t = datetime.now()
            form_res =  format_res(osm_results)
            form_rej = format_res(all_reject)
            update_timestats("format_res", t)
            
            return {"match": form_res, "rejected": form_rej }
    
    return {"rejected": format_res(all_reject)}


# In[ ]:


def process_addresses(to_process_addresses, check_results=True, osm_structured=False, with_extra_house_number=True):
    
    all_reject = pd.DataFrame()
    osm_addresses        = pd.DataFrame()
    rejected_addresses   = pd.DataFrame()
    
    chunk = to_process_addresses.copy()
    
    for transformers in transformers_sequence:
        vlog ("--------------------------")
        log(f"Transformers ({chunk.shape[0]:3} records): " + ";".join(transformers))
        vlog ("--------------------------")

        try :
            osm_results, rejected, step_stats = transform_and_process(chunk, transformers, addr_key_field, 
                                                                      street_field=street_field, housenbr_field=housenbr_field, 
                                                                      postcode_field=postcode_field, city_field=city_field,
                                                                      country_field=country_field,
                                                                      check_results=check_results,
                                                                      osm_structured=osm_structured)
            
            osm_addresses =      osm_addresses.append(osm_results, sort=False).drop_duplicates()
            rejected_addresses = rejected_addresses.append(rejected, sort=False).drop_duplicates()
            
        except Exception as e: 
            osm_results = chunk[[addr_key_field]]
            osm_results["method"] = "error on " + ";".join(transformers)
            osm_addresses =      osm_addresses.append(osm_results, sort=False).drop_duplicates()
            
            log(f"Error during processing : {e}")
            traceback.print_exc(file=sys.stdout)
#             return {"error": str(e)}
        
        chunk  = chunk[~chunk[addr_key_field].isin(osm_results[addr_key_field])].copy()
        
        #all_reject = all_reject.append(rejected, sort=False)
        if chunk.shape[0]==0:
            break
            
        
        vlog(step_stats)
    if with_extra_house_number and osm_addresses.shape[0] > 0:
        osm_addresses = add_extra_house_number(osm_addresses, to_process_addresses, 
                                               street_field=street_field, housenbr_field=housenbr_field,
                                               postcode_field=postcode_field, city_field=city_field)
          
    #log(osm_addresses.method.value_counts())
    return osm_addresses, rejected_addresses #{"match": format_res(osm_results), "rejected": format_res(all_reject)}
    
#     return pd.DataFrame()


# In[ ]:


def get_arg(argname, def_val):
    if argname in request.form: 
        return request.form[argname]
    return request.args.get(argname, def_val)


import sys

app = Flask(__name__)

@app.route('/search/', methods=['GET', 'POST'])
def search():
#     print("search!")
    t = datetime.now()
    
    for k in AddressCleanserUtils.timestats:
        AddressCleanserUtils.timestats[k]=timedelta(0)
        
    data= {street_field   : get_arg("street",      ""),
           housenbr_field : get_arg("housenumber", ""),
           city_field     : get_arg("city",        ""),
           postcode_field : get_arg("postcode",    ""),
           country_field  : get_arg("country",     ""),

          }
    no_reject = get_arg("no_reject", False)
    
    if get_arg("check_result", "yes") == "no":
        check_results = False
        log("Won't check OSM results")
    else:
        check_results = True
        log("Will check OSM results")
    
    if get_arg("struct_osm", "no") == "no":
        osm_structured = False
        log("Will call unstructured OSM")
    else:
        osm_structured = True
        log("Will call structured OSM")
    
    if get_arg("extra_house_nbr", "yes") == "no":
        with_extra_house_number = False
        vlog("Will skip extra house nbr")
    else:
        with_extra_house_number = True
        vlog("Will do extra house nbr")
    
#     update_timestats("check_args", t)
    
    res = process_address(data,
                          check_results=check_results, 
                          osm_structured=osm_structured, 
                          with_extra_house_number= with_extra_house_number)
    
    log(f"Input: {data}")
    log(f"Result: {res}")
    log(f"no_reject: {no_reject}")
   
    update_timestats("global", t)
    

    if with_timing_info: 
        res["timing"] = {k: AddressCleanserUtils.timestats[k].total_seconds()*1000 for k in AddressCleanserUtils.timestats}
    
    if no_reject != False:
        del res["rejected"]
    
    
    return jsonify(res)


# In[26]:


# # data_batch= pd.DataFrame(columns = [street_field, housenbr_field, city_field,postcode_field,country_field])
# @app.route('/search_async/', methods=['GET', 'POST'])
# def search_async():
#     global data_batch
# #     print("search!")
#     t = datetime.now()
    
#     for k in AddressCleanserUtils.timestats:
#         AddressCleanserUtils.timestats[k]=timedelta(0)
        
#     data= {street_field   : get_arg("street",      ""),
#            housenbr_field : get_arg("housenumber", ""),
#            city_field     : get_arg("city",        ""),
#            postcode_field : get_arg("postcode",    ""),
#            country_field  : get_arg("country",     ""),
#            "time":          datetime.now()
#           }
    
#     data_batch = pd.concat([data_batch, pd.DataFrame([data])])
    
#     log(data_batch)
    
#     return jsonify({"key": hash(str(data))})


# In[23]:


# data_batch = pd.DataFrame(columns = ["street_field", "housenbr_field", "city_field","postcode_field","country_field"])
# data = {"street_field": "a", "housenbr_field": "b", "city_field": "c","postcode_field":"d","country_field":"d"}
# pd.concat([data_batch, pd.DataFrame([data])])


# In[24]:


# data_batch


# In[3]:



@app.route('/test/', methods=['GET', 'POST'])
def test():
#     print("search!")
    start = datetime.now()
    timings = {}
#     for k in AddressCleanserUtils.timestats:
#         AddressCleanserUtils.timestats[k]=timedelta(0)
        
    data= {street_field   : get_arg("street",      ""),
           housenbr_field : get_arg("housenumber", ""),
           city_field     : get_arg("city",        ""),
           postcode_field : get_arg("postcode",    ""),
           country_field  : get_arg("country",     ""),

          }
    no_reject = get_arg("no_reject", False)
    
    if get_arg("check_result", "yes") == "no":
        check_results = False
        log("Won't check OSM results")
    else:
        check_results = True
        log("Will check OSM results")
    
    if get_arg("struct_osm", "no") == "no":
        osm_structured = False
        log("Will call unstructured OSM")
    else:
        osm_structured = True
        log("Will call structured OSM")
    
    if get_arg("extra_house_nbr", "yes") == "no":
        with_extra_house_number = False
        vlog("Will skip extra house nbr")
    else:
        with_extra_house_number = True
        vlog("Will do extra house nbr")
    
    timings["start"]= (datetime.now() - start).total_seconds() * 1000 

    to_process_addresses = get_init_df(data)
    
    timings["got init df"]= (datetime.now() - start).total_seconds() * 1000 
    

    to_process_addresses[osm_addr_field] = to_process_addresses[street_field  ].fillna("") + ", "+                                            to_process_addresses[housenbr_field].fillna("") + ", "+                                            to_process_addresses[postcode_field].fillna("") + " " +                                           to_process_addresses[city_field    ].fillna("") + ", "+                                           to_process_addresses[country_field ].fillna("") 
    
    to_process_addresses["accept_language"] = "fr"
    to_process_addresses[osm_addr_field]= to_process_addresses[osm_addr_field].str.replace("^[ ,]+", "")

    timings["df ready"]= (datetime.now() - start).total_seconds() * 1000 

    with_lambda = False
    
    if with_lambda: 
        osm_res_field = "osm"
        to_process_addresses[osm_res_field] = to_process_addresses[[osm_addr_field, "accept_language"]].apply(lambda row: get_osm(row[osm_addr_field], row["accept_language"]), axis=1)

        timings["called lambda"]= (datetime.now() - start).total_seconds() * 1000 
        res = to_process_addresses[osm_res_field].iloc[0][0]
    else: 
        res= get_osm(to_process_addresses[osm_addr_field].iloc[0], "fr")
        timings["called direct"]= (datetime.now() - start).total_seconds() * 1000 

    res = {"res": res,
           "timing": timings}

    return jsonify(res)


# In[ ]:


def remove_empty_values(dct_lst):
    # Remove empty values in a list of dict
    return [{k: v for k, v in item.items() if not pd.isnull(v) and v != ""} for item in dct_lst]


# In[5]:


# Call to this : curl -F media=@address_sample100.csv http://127.0.0.1:5000/batch/ -X POST -F mode=long

@app.route('/batch/', methods=['POST'])
def batch():
    log("batch")
    
   
    mode = "short"
    if "mode" in request.form :
        mode = request.form["mode"]
        if not mode in ["geo", "short", "long"]:
            return f'[{{"error": "Invalid mode {mode}"}}]'
        
    with_reject = False
    if "with_rejected" in request.form:
        if request.form["with_rejected"] == "yes":
            with_reject = True
            log("Will return rejects")
        elif request.form["with_rejected"] != "no":
            return f'[{{"error": "Invalid with_rejected {request.form["with_rejected"]}"}}]'
#           return ({"error": f"Invalid with_rejected value : {request.form['with_rejected']}"})

    if get_arg("check_result", "yes") == "no":
        check_results = False
        log("Won't check OSM results")
    else:
        check_results = True
        log("Will check OSM results")
        
    if get_arg("struct_osm", "no") == "no":
        osm_structured = False
        log("Will call unstructured OSM")
    else:
        osm_structured = True
        log("Will call structured OSM")

    if get_arg("extra_house_nbr", "yes") == "no":
        with_extra_house_number = False
        vlog("Will skip extra house nbr")
    else:
        with_extra_house_number = True
        vlog("Will do extra house nbr")
    
        
        
    key_name = (list(request.files.keys())[0])
    
    #print(request.files[0])
    
    df = pd.read_csv(request.files[key_name], dtype=str)
    log("Input: \n" + df.to_string(max_rows=10))
    
    mandatory_fields = [street_field, housenbr_field , postcode_field , city_field, country_field, addr_key_field]
    for field in mandatory_fields:
        if field not in df: 
            return f"[{{\"error\": \"Field \'{field}\' mandatory in file. All mandatory fields are {';'.join(mandatory_fields)}\"}}]"
    
    res, rejected_addresses = process_addresses(df,
                                                check_results=check_results, 
                                                osm_structured=osm_structured,
                                                with_extra_house_number= with_extra_house_number)
    
    
    if type(res) == dict :
        return {0:res}

    
    if res is None or res.shape[0] == 0:
        return '[]'
        
    try: 
        if mode == "geo":
            res = res[[addr_key_field,"lat", "lon", "place_rank", "method"]]
        elif mode == "short":
            res = df.merge(res)[[addr_key_field,
                       "lat", "lon", "place_rank", "method", "place_id",
                       "addr_out_street", "addr_out_number", "in_house_nbr", "lpost_house_nbr", "lpost_unit", "addr_out_postcode", "addr_out_city",   "addr_out_country" ]]
        elif mode == "long":
            res = df.merge(res)
            
       
        if with_reject:
            rejected_rec = rejected_addresses.groupby(addr_key_field).apply(lambda rec: remove_empty_values(rec.to_dict(orient="records")) ).rename("rejected").reset_index()
            res = res.merge(rejected_rec, how="outer")


        res["lat"] = res["lat"].astype(float)
        res["lon"] = res["lon"].astype(float)
    except KeyError as e:
        log(f"Error during column selection: {e}")
        traceback.print_exc(file=sys.stdout)
        
    log("Output: \n"+res.iloc[:, 0:9].to_string(max_rows=9))
    
    return res.to_json(orient="records")
    #request.files:


# In[ ]:


#! jupyter nbconvert --to python AddressCleanserREST.ipynb


# In[ ]:




