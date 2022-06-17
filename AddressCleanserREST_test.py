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


# In[4]:


from config_batch import * 


# # Functions

# In[5]:


ws_hostname = "127.0.1.1"
ws_hostname = "10.1.0.45"

# ws_hostname = "192.168.1.3"


# In[6]:


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
            res["time"] = datetime.now() - t
            return res
    except Exception as e:
        return str(e)
    


# In[25]:


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


# In[8]:


def expand_json(addresses):
    addresses["status"]= addresses.json.apply(lambda d: "error" if "error" in d else "match" if "match" in d else "rejected")
    addresses["time"]  = addresses.json.apply(lambda d: d["time"])

    addresses["timing"]  = addresses.json.apply(lambda d: d["timing"] if "timing" in d else {})

    addresses["method"]= addresses.json.apply(lambda d: d["match"][0]["method"] if len(d)>0 and "match" in d else "none")
    
    for field in ["street", "number", "postcode", "city"]:
        addresses[field]= addresses.json.apply(lambda d: d["match"][0]["addr_out_"+field] if len(d)>0 and "match" in d else "")
    return 


# # Calls

# ## Single address calls

# In[9]:


call_ws({street_field:   "Av. Fonsny", 
         housenbr_field: "20 bus 22",
         city_field:     "Saint-Gilles",
         postcode_field: "1060",
         country_field:  "Belgium"}, check_result=True, structured_osm=False)


# In[ ]:


call_ws({street_field:   "", 
         housenbr_field: "",
         city_field:     "Dinant",
         postcode_field: "5500",
         country_field:  "Belgium"}, check_result=True, structured_osm=True)


# In[ ]:





# In[ ]:





# ## Batch calls (row by row)

# In[136]:


addresses = get_addresses("address.csv.gz")
addresses = addresses.sample(1000).copy()


# ### Simple way

# In[149]:


addresses["json"] = addresses.progress_apply(call_ws, check_result=False, structured_osm=False, axis=1)


# In[ ]:


# addresses


# ### Using Dask

# In[ ]:


dd_addresses = dd.from_pandas(addresses, npartitions=4)

dask_task = dd_addresses.apply(call_ws, meta=('x', 'str'), axis=1)

with ProgressBar(): 
    addresses["json"] = dask_task.compute()


# In[ ]:


expand_json(addresses)


# In[ ]:


addresses


# ## Batch calls (batch WS)

# In[10]:


addresses = pd.read_csv(f"../GISAnalytics/data/geocoding/kbo_1000_sample.csv")
addresses = addresses.rename(columns={"Unnamed: 0": addr_key_field, "address": street_field})
addresses[city_field] = ""
addresses[country_field] =  "Belgique"
addresses[housenbr_field] = ""
addresses[postcode_field]=""
addresses


# In[137]:


addresses


# ### Single block

# In[26]:


# Only geocoding
# addresses["StreetFR"] = ""
call_ws_batch(addresses, mode="geo", check_result=True, structured_osm=True)


# In[144]:


# Geocode + address
call_ws_batch(addresses, mode="short") 


# In[ ]:


# Geocode + address, with rejected addresses
call_ws_batch(addresses, mode="long", with_reject=True)


# In[ ]:


# call_ws_batch(addresses[addresses.EntityNumber.str.startswith("0554.81")], mode="long", with_reject=True)


# In[ ]:


# a[a.in_house_nbr.str.upper() != a.lpost_house_nbr.str.upper()]


# ### Batch blocs

# In[ ]:


def call_ws_batch_chunks(addr_data, mode="geo", with_reject=False, check_result=True, structured_osm=False, chunk_size=100): 
    ## TODO : find a better way with dask? It seems that map_partitions does not support function returning dataframes. 

    chunks = np.array_split(addr_data, addr_data.shape[0]//chunk_size)

    res= [call_ws_batch(chunk, mode=mode, 
                        check_result=check_result, 
                        structured_osm=structured_osm) for chunk in tqdm(chunks)]
    df_res = pd.concat(res, sort=False)
    return df_res


# In[146]:


df_res = call_ws_batch_chunks(addresses, chunk_size=100, mode="short", check_result=False)
df_res


# In[129]:


df_res[df_res.method=="nonum"].sort_values("postcode")


# In[ ]:


df_res[df_res.in_house_nbr.str.upper() != df_res.lpost_house_nbr.str.upper()]


# In[ ]:


# df_res[df_res.addr_out_number.str.upper() != df_res.lpost_house_nbr.str.upper()]


# In[ ]:


df_res.method.value_counts()


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


# In[ ]:




