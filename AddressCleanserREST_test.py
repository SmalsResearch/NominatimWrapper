#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import urllib

import numpy as np

import json

from tqdm.autonotebook import tqdm

#%matplotlib inline

tqdm.pandas()

import dask.dataframe as dd

from dask.multiprocessing import get
from dask.diagnostics import ProgressBar

from datetime import datetime
import matplotlib.pyplot as plt

from IPython.display import display


# In[2]:


import urllib3


# In[3]:


http = urllib3.PoolManager()


# In[5]:


from config_batch import * 


# # Functions

# In[6]:


ws_hostname = "127.0.1.1"
ws_hostname = "172.27.0.64"

# ws_hostname = "192.168.1.3"


# In[7]:


def call_ws(addr_data, check_result=True, structured_osm=False): #lg = "en,fr,nl"
    t = datetime.now()
    
    params = urllib.parse.urlencode({"street": addr_data[street_field],
                                     "housenumber": addr_data[housenbr_field],
                                     "city": addr_data[city_field],
                                     "postcode": addr_data[postcode_field],
                                     "country": addr_data[country_field],
                                     "check_result" : "yes" if check_result else "no",
                                     "struct_osm" : "yes" if structured_osm else "no"
                                    })
    url = f"http://{ws_hostname}:5000/search/?{params}"
    
#     print(url)
    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             print(res)
            res["time"] = (datetime.now() - t).total_seconds()
            return res
    except Exception as e:
        return str(e)
    


# In[9]:


def call_ws_batch(addr_data, mode="geo", with_reject=False, check_result=True, structured_osm=False): #lg = "en,fr,nl"
#     print(addr_data)
#     print(addr_data.shape)
#     print()
    file_data = addr_data.rename(columns = {
        street_field : "street",
        housenbr_field: "housenumber",
        postcode_field: "postcode",
        city_field: "city",
        country_field: "country",
        addr_key_field : "addr_key"
    }).to_csv(index=False)
    
    r = http.request(
    'POST',
    f'http://{ws_hostname}:5000/batch',
    fields= { 
        'media': ('addresses.csv', file_data),
        'mode': mode,
        "with_rejected" : "yes" if with_reject else "no",
        "check_result" : "yes" if check_result else "no",
        "struct_osm" : "yes" if structured_osm else "no",
        #"extra_house_nbr": "no"
    })
    
    try:
        res = pd.DataFrame(json.loads(r.data.decode('utf-8')))
    except ValueError:
        print("Cannot decode result:")
        print(r.data.decode('utf-8'))
        print(json.loads(r.data.decode('utf-8')))
        return 
#     display(res)
    return res


# In[62]:


def expand_json(addresses):
    addresses["status"]= addresses.json.apply(lambda d: "error" if "error" in d else "match" if "match" in d else "rejected")
    addresses["time"]  = addresses.json.apply(lambda d: d["time"])

    addresses["timing"]  = addresses.json.apply(lambda d: d["timing"] if "timing" in d else {})

    addresses["method"]= addresses.json.apply(lambda d: d["match"][0]["method"] if len(d)>0 and "match" in d else "none")
    
    for field in ["street", "number", "postcode", "city"]:
        addresses["addr_out_"+field]= addresses.json.apply(lambda d: d["match"][0]["addr_out_"+field] if len(d)>0 and "match" in d else "")
    return 


# # Calls

# ## Single address calls

# In[379]:


res=call_ws({street_field:   "Av. Fonsny",          housenbr_field: "20",         city_field:     "Saint-Gilles",         postcode_field: "1060",         country_field:  "Belgium"}, check_result=False, structured_osm=False)
res


# ## Batch calls (row by row)

# In[375]:


# KBO dataset
addresses = get_addresses("address.csv.gz")
# addresses = addresses.sample(10000, replace=True).copy()
addresses["addr_key"] = addresses["addr_key"].astype(str)
addresses


# In[264]:


addresses = get_addresses("../GISAnalytics/data/geocoding/resto_1000_sample.csv.gz")
# addresses = get_addresses("../GISAnalytics/data/geocoding/best_1000_sample.csv.gz")
addresses["addr_key"] = addresses.index.astype(str)
addresses


# In[349]:


addresses = pd.concat([
    get_addresses("../GISAnalytics/data/geocoding/resto_1000_sample.csv.gz"),
    get_addresses("../GISAnalytics/data/geocoding/best_1000_sample.csv.gz"),
    get_addresses("address.csv.gz")])
addresses = addresses.reset_index(drop=True)
addresses["addr_key"] = addresses.index.astype(str)


# In[312]:


# addresses = addresses[addresses.addr_key.isin([ "2078829"])]#"1622",
# addresses


# 
# ### Simple way

# In[380]:


addresses_seq = addresses.copy()

t = datetime.now()
addresses_seq["json"] = addresses_seq.fillna("").progress_apply(call_ws, check_result=False, structured_osm=False, axis=1)
tot_time = (datetime.now() - t).total_seconds()
print(f"{tot_time:.2f} seconds, {addresses_seq.shape[0]/tot_time:.2f} it/s")
# KBO dataset:
# Normal mode: 128.78 seconds, 7.77 it/s
# Fastmode:     68.80 seconds, 14.54 it/s

#Resto dataset: 
# Normal mode: 145.73 seconds, 6.86 it/s
# Fast mode:    82.99 seconds, 12.05 it/s

# Best dataset:
# Normal mode: 108.53 seconds, 9.21 it/s
# Fast mode: 37.44 seconds, 26.71 it/s


# In[315]:


expand_json(addresses_seq)
addresses_seq


# In[392]:


addresses_seq.json.apply(lambda json_rec: [ r["dist_to_match"] for r in json_rec['reject']] if "reject" in json_rec else []).iloc[0:60]


# In[394]:


addresses_seq.iloc[6].json


# In[150]:


addresses_seq.method.value_counts()


# ### Using Dask

# In[229]:


addresses_dask = addresses.copy()


# In[372]:


t = datetime.now()
dd_addresses = dd.from_pandas(addresses_dask, npartitions=8)

dask_task = dd_addresses.apply(call_ws, check_result=False, meta=('x', 'str'), axis=1)

with ProgressBar(): 
    addresses_dask["json"] = dask_task.compute()
    
tot_time = (datetime.now() - t).total_seconds()
print(f"{tot_time:.2f} seconds, {addresses_dask.shape[0]/tot_time:.2f} it/s")
# KBO dataset:
# Normal mode: 24.52 seconds, 40.79 it/s
# Fastmode:    15.81 seconds, 63.27 it/s


# Resto dataset:
# Normal mode: 27.86 seconds, 35.89 it/s
# Fast mode:   18.44 seconds, 54.23 it/s

# Best dataset: 
# Normal mode: 16.11 seconds, 62.07 it/s
# Fast mode:    9.76 seconds, 102.42 it/s


# In[90]:


# 1000, 1 worker: 4m18
# 4 workers, npart=4 : 1m20
# 8 workers, npart=4 : 1m20
# 8 workers, npart=8 : 44s

# with checker=False:
# 8 workers, npart=8 : 24s


# In[370]:


expand_json(addresses_dask)
addresses_dask


# In[201]:


addresses_dask.method.value_counts()#.json.loc[550]


# In[232]:


mg = addresses_seq[["addr_key", "city", "postcode","street", "housenumber", "addr_out_street", "addr_out_number", "addr_out_postcode", "addr_out_city"]].merge(
    addresses_dask[["addr_key", "city", "postcode","street", "housenumber", "addr_out_street", "addr_out_number", "addr_out_postcode", "addr_out_city"]], how="outer", indicator=True)
if mg.shape[0] == addresses.shape[0]:
    print("Same result in seq and dask run!")
else: 
    print("!!! Not the same result in seq and dask run!")
    


# In[233]:


mg[mg._merge != "both"].sort_values("addr_key")


# ## Batch calls (batch WS)

# ### Single block

# In[347]:


t = datetime.now()

addresses_batch = call_ws_batch(addresses[[addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field]], 
                                mode="long", 
                                check_result=False, 
                                structured_osm=False)

tot_time = (datetime.now() - t).total_seconds()
print(f"{tot_time:.2f} seconds, {addresses.shape[0]/tot_time:.2f} it/s")
# KBO dataset: 33.94 seconds, 29.46 it/s
# Best:        24.99 seconds, 40.01 it/s
# Resto:       38.33 seconds, 26.09 it/s


# In[344]:


addresses


# In[242]:


mg = addresses_seq[[ "city", "postcode","street", "housenumber", "method", "addr_out_street", "addr_out_number", "addr_out_postcode", "addr_out_city", "addr_key"]].fillna("").replace("fast", "orig").merge(
    addresses_batch[["city", "postcode","street", "housenumber", "method", "addr_out_street", "addr_out_number", "addr_out_postcode", "addr_out_city", "addr_key"]].fillna(""), how="outer", indicator=True)
if mg[mg._merge == "both"].shape[0] == addresses.shape[0]:
    print("Same result in seq and dask run!")
else: 
    print("!!! Not the same result in seq and dask run!")
    


# In[243]:


mg[mg._merge != "both"]


# In[213]:


# Geocode + address
call_ws_batch(addresses[[addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field]], mode="geo", check_result=False) 


# In[249]:


# Geocode + address
call_ws_batch(addresses[[addr_key_field, street_field, housenbr_field, postcode_field, city_field, country_field]], mode="short", check_result=False) 


# In[360]:


# Geocode + address, with rejected addresses
call_ws_batch(addresses, mode="long", with_reject=True)


# ### Batch blocs

# In[359]:


# addresses = addresses.sample(10000, replace=True)
# addresses = addresses.reset_index(drop=True)
# addresses["addr_key"]= addresses.index.astype(str)


# In[352]:


t = datetime.now()

nb_threads=8

chunks = np.array_split(addresses, nb_threads) # addresses.shape[0]//100)

print(f"{len(chunks)} chunks on {nb_threads} threads")

import multiprocess as mp

p = mp.Pool(nb_threads)

def f(chunk):
    return call_ws_batch(chunk, mode="long", 
                        check_result=False, 
                        structured_osm=False)

with p:
     res= list(tqdm(p.imap(f, chunks), total=len(chunks)))
    
addresses_batch2 = pd.concat(res).reset_index(drop=True)

tot_time = (datetime.now() - t).total_seconds()
print(f"{tot_time:.2f} seconds, {addresses.shape[0]/tot_time:.2f} it/s")
# KBO:    9.28 seconds, 107.72 it/s
# Best:   6.88 seconds, 145.43 it/s
# Resto: 11.79 seconds,  84.85 it/s


# In[361]:


# addresses_batch2


# In[317]:


mg = addresses_seq[[ "city", "postcode","street", "housenumber", "method", "addr_out_street", "addr_out_number", "addr_out_postcode", "addr_out_city", "addr_key"]].fillna("").replace("fast", "orig").merge(
    addresses_batch2[["city", "postcode","street", "housenumber", "method", "addr_out_street", "addr_out_number", "addr_out_postcode", "addr_out_city", "addr_key"]].fillna(""), how="outer", indicator=True)
if mg[mg._merge == "both"].shape[0] == addresses.shape[0]:
    print("Same result in seq and dask run!")
else: 
    print("!!! Not the same result in seq and dask run!")
    


# In[318]:


mg[mg._merge != "both"].sort_values("addr_key")


# ## Comparing options

# In[ ]:


addresses = get_addresses("address.csv.gz")
addresses = addresses[addresses[country_field] == "Belgique"]
addresses = addresses.sample(10000).copy()


# In[ ]:


results = {}
it_per_seconds=pd.DataFrame()

for check_label in ["check", "nocheck"]:
    for struct_label in ["struct", "unstruct" ]:
        print(check_label, struct_label)
        start=datetime.now()
        
        results[(check_label, struct_label)] = call_ws_batch_chunks(addresses, 
                                                                    mode="short", 
                                                                    check_result   =  check_label == "check", 
                                                                    structured_osm =  struct_label == "struct")
        
        it_per_seconds.loc[check_label, struct_label] = addresses.shape[0] / (datetime.now()-start).total_seconds()
print("Iterations per seconds:")
it_per_seconds


# In[ ]:


print("Match rate")
pd.DataFrame({k1: {k2: results[(k1,k2)].shape[0]/addresses.shape[0] for k2 in ["struct", "unstruct"]} 
                  for k1 in  ["check","nocheck"]})


# In[ ]:


print("Match rate (without nostreet)")
pd.DataFrame({k1: {k2: results[(k1,k2)].query("method!='nostreet'").shape[0]/addresses.shape[0] for k2 in ["struct", "unstruct"]} 
                  for k1 in  ["check","nocheck"]})


# In[ ]:


print("Unmatched addresses")
for k1 in results:
    print(k1)
    nomatch=addresses[~addresses[addr_key_field].isin(results[k1]["addr_key"])]
    display(nomatch)
    print(nomatch[country_field].value_counts())


# In[ ]:


vc_values = pd.DataFrame(columns=results.keys(), index=results.keys())

for k1 in results:
    vc_values.loc[k1, k1] = results[k1].shape[0]
    for k2 in results:
        if k1>k2:
            r1=results[k1]
            r2=results[k2]
            mg = r1[["addr_key", "place_id"]].merge(r2[["addr_key", "place_id"]], on="addr_key", how="outer", indicator=True)
 
            vc = mg._merge.value_counts()

            mismatches = mg[mg.place_id_x != mg.place_id_y][["addr_key"]]
            mismatches = mismatches.merge(addresses.rename({addr_key_field:"addr_key"}, axis=1))
            mismatches = mismatches.merge(r1[["addr_key", "addr_out_street", "addr_out_number", "extra_house_nbr", "addr_out_postcode", "addr_out_city"]], on="addr_key")
            mismatches = mismatches.merge(r2[["addr_key", "addr_out_street", "addr_out_number", "extra_house_nbr", "addr_out_postcode", "addr_out_city"]], on="addr_key")
            mismatches.columns = pd.MultiIndex.from_arrays([["Input"]*6 + [f"x:{k1}"]*5 + [f"y:{k2}"]*5, mismatches.columns])

            mismatch_values = mismatches[(mismatches[f"x:{k1}"].rename(lambda x: x.replace("_x", ""), axis=1).fillna("") != 
                                          mismatches[f"y:{k2}"].rename(lambda x: x.replace("_y", ""), axis=1).fillna("")).any(axis=1)]
            
            mismatch_values_no_nmbr = mismatches[(mismatches[f"x:{k1}"].rename(lambda x: x.replace("_x", ""), axis=1).drop("addr_out_number", axis=1).fillna("") != 
                                                  mismatches[f"y:{k2}"].rename(lambda x: x.replace("_y", ""), axis=1).drop("addr_out_number", axis=1).fillna("")).any(axis=1)]
            
            
            vc_label = f"{vc['both']} ({mismatches.shape[0]} - {mismatch_values.shape[0]} - {mismatch_values_no_nmbr.shape[0]}) / {vc['left_only']} / {vc['right_only']}"
            vc_values.loc[k1, k2]=vc_label

                
            print(f"{k1} vs {k2}")
            print(vc_label)
            print("-----------------------------")
            
            print(f"Only in {k1}")
            display(r1[r1.addr_key.isin(mg[mg._merge=="left_only"].addr_key)].merge(addresses.rename({addr_key_field:"addr_key"}, axis=1)))
            
            print(f"Only in {k2}")
            display(r2[r2.addr_key.isin(mg[mg._merge=="right_only"].addr_key)].merge(addresses.rename({addr_key_field:"addr_key"}, axis=1)))
            
            print("Mismatch on place_id")
            display(mismatches)
            
            print("Mismatch on values")
            
            display(mismatch_values)
            
            print("Mismatch on values (no nbr)")
            display(mismatch_values_no_nmbr)
            
            print("#######################")
            
# display(vc_values.fillna(""))


# In[ ]:


print("Common in both (disagree on place_id - disagree on values - disagree on values, ignoring number) / results only for row / results only for columns")
vc_values.fillna("")


# # tests

# In[100]:


osm_host ="172.27.0.64:8080"
def get_osm(addr, accept_language = ""): #lg = "en,fr,nl"
    params = urllib.parse.urlencode({"q": addr,
                                    "format":"jsonv2",
                                    "accept-language":accept_language,
                                    "addressdetails":"1",
                                    "namedetails" : "1",
                                    "limit": "50"
                                    })
    
    url = "http://%s/search.php?%s"%(osm_host, params)
#     print(f"Call to OSM: {url}")
    try: 
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             return res
            return [ {field: item[field] for field in ["place_id", "lat", "lon", "display_name", "address", "namedetails", "place_rank", "category", "type"]} for item in res] 
    except Exception as e:
        raise Exception (f"Cannot get OSM results ({osm_host}): {e}") 


# In[322]:


get_ipython().run_line_magic('timeit', 'get_osm("Av. Fonsny 20, 1060 Bruxelles")')


# In[323]:


get_ipython().run_line_magic('timeit', 'call_ws_test({street_field:   "Av. Fonsny",          housenbr_field: "20",         city_field:     "Saint-Gilles",         postcode_field: "1060",         country_field:  "Belgium"}, check_result=False, structured_osm=False)')
# res

