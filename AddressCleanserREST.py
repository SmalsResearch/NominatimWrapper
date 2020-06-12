#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask,  request,jsonify

import pandas as pd

import os

from  importlib import reload
import AddressCleanserUtils
reload(AddressCleanserUtils)
from AddressCleanserUtils import *

from IPython.display import display


import sys, traceback

from datetime import datetime, timedelta


# In[ ]:


import config_REST
reload(config_REST)
from config_REST import *


# In[ ]:





# In[ ]:


# TO RUN : 
# jupyter nbconvert --to python AddressCleanserREST.ipynb
# export  FLASK_APP=AddressCleanserREST.py ; export  FLASK_ENV=development ;  flask   run  

# OR : 
# gunicorn -w 8 -b 127.0.0.1:5000 wsgi:app


# In[2]:


# !jupyter nbconvert --to python AddressCleanserREST.ipynb


# In[1]:


# AddressCleanserUtils.pbar.unregister()

AddressCleanserUtils.with_dask         = False
AddressCleanserUtils.check_osm_results = True

AddressCleanserUtils.addr_key_field  = addr_key_field

AddressCleanserUtils.regex_replacements = regex_replacements

AddressCleanserUtils.use_osm_parent = use_osm_parent 


# In[6]:


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
        to_copy_field = ["osm_id", "place_id", "lat","lon","display_name", "place_rank", "method", "extra_house_nbr"] + list(collapse_params.keys())  + list(filter(lambda x: x.startswith("SIM"), row.index))
        res =  {}

        for f in to_copy_field:
            if f in row: 
                res[f] = row[f]

        return res


# In[ ]:


def format_res(res):
    return list(res.fillna("").apply(lambda row: get_row_dict(row, False), axis=1))


# In[1]:


transformers_sequence = [ ["orig"],
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

def process_address(data):
    vlog(f"Will process {data}")
    to_process_addresses = get_init_df(data)
    
    vlog("Got dataframe")
    all_reject = pd.DataFrame()
    for transformers in transformers_sequence:
        vlog ("--------------------------")
        vlog("| Transformers : " + ";".join(transformers))
        vlog ("--------------------------")

        try :
            osm_results, rejected, step_stats = transform_and_process(to_process_addresses, transformers, addr_key_field, 
                                                                      street_field=street_field, housenbr_field=housenbr_field, 
                                                                      postcode_field=postcode_field, city_field=city_field,
                                                                      country_field=country_field)
            
        except Exception as e: 
            log(f"Error during processing : {e}")
            traceback.print_exc(file=sys.stdout)
            return {"error": str(e)}
        
        all_reject = all_reject.append(rejected, sort=False)
        
        vlog(step_stats)
        if osm_results.shape[0] > 0:
            osm_results = add_extra_house_number(osm_results, to_process_addresses, street_field=street_field, housenbr_field=housenbr_field)
            
            return {"match": format_res(osm_results), "rejected": format_res(all_reject)}
    
    return {"rejected": format_res(all_reject)}


# In[ ]:


def process_addresses(to_process_addresses):
    
    all_reject = pd.DataFrame()
    for transformers in transformers_sequence:
        vlog ("--------------------------")
        vlog("| Transformers : " + ";".join(transformers))
        vlog ("--------------------------")

        try :
            osm_results, rejected, step_stats = transform_and_process(to_process_addresses, transformers, addr_key_field, 
                                                                      street_field=street_field, housenbr_field=housenbr_field, 
                                                                      postcode_field=postcode_field, city_field=city_field,
                                                                      country_field=country_field)
        except Exception as e: 
            log(f"Error during processing : {e}")
            traceback.print_exc(file=sys.stdout)
            return {"error": str(e)}
        
        all_reject = all_reject.append(rejected, sort=False)
        
        vlog(step_stats)
        if osm_results.shape[0] > 0:
            osm_results = add_extra_house_number(osm_results, to_process_addresses, street_field=street_field, housenbr_field=housenbr_field)
            
            return osm_results #{"match": format_res(osm_results), "rejected": format_res(all_reject)}
    
    return pd.DataFrame()


# In[2]:


def get_arg(argname, def_val):
    if argname in request.form: 
        return request.form[argname]
    return request.args.get(argname, def_val)


import sys

app = Flask(__name__)

@app.route('/search/', methods=['GET', 'POST'])
def search():
#     print("search!")

    for k in AddressCleanserUtils.timestats:
        AddressCleanserUtils.timestats[k]=timedelta(0)
        
    data= {street_field   : get_arg("street", ""),
           housenbr_field : get_arg("housenumber", ""),
           city_field     : get_arg("city", ""),
           postcode_field : get_arg("postcode", ""),
           country_field  : get_arg("country", ""),

          }
    no_reject = get_arg("noreject", False)
   
    res = process_address(data)
    log(f"Input: {data}")
    log(f"Result: {res}")
    
    if with_timing_info: 
        res["timing"] = {k: AddressCleanserUtils.timestats[k].total_seconds() for k in AddressCleanserUtils.timestats}
    
    if no_reject != False:
        del res["rejected"]
        
    return jsonify(res)




# In[ ]:


# Call to this : curl -F media=@address_sample100.csv http://127.0.0.1:5000/batch/ -X POST -F mode=long

@app.route('/batch/', methods=['POST'])
def batch():
    log("batch")
    
    mode = "short"
    if "mode" in request.form :
        mode = request.form["mode"]

    key_name = (list(request.files.keys())[0])
    
    #print(request.files[0])
    
    df = pd.read_csv(request.files[key_name], dtype=str)
    log(df)
    
    mandatory_fields = [street_field, housenbr_field , postcode_field , city_field, country_field, addr_key_field]
    for field in mandatory_fields:
        if field not in df: 
            return f"Field '{field} mandatory in file. All mandatory fields are {mandatory_fields}\n"
    
    res = process_addresses(df)
    
    
    if res is None or res.shape[0] == 0:
        return '[]'
        
    if mode == "geo":
        res = res[[addr_key_field,"lat", "lon", "place_rank"]]
    elif mode == "short":
        res = df.merge(res)[[addr_key_field,
                   "lat", "lon", "place_rank", 
                   "addr_out_street", "addr_out_number", "extra_house_nbr", "addr_out_postcode", "addr_out_city",   "addr_out_country" ]]
    elif mode == "long":
        res = df.merge(res)
    else:
        return f"Invalid mode {mode}"
    
    res["lat"] = res["lat"].astype(float)
    res["lon"] = res["lon"].astype(float)
    
    log(res)
    return res.to_json(orient="records")
    #request.files:


# In[3]:


#! jupyter nbconvert --to python AddressCleanserREST.ipynb

