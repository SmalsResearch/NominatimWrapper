#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import urllib

import numpy as np

import json

from tqdm.autonotebook import  tqdm

#%matplotlib inline

tqdm.pandas(tqdm)


# import jellyfish
import dask.dataframe as dd

# from dask.multiprocessing import get


from importlib import reload

import AddressCleanserUtils
reload(AddressCleanserUtils)
from AddressCleanserUtils import *

# import multiprocessing
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# In[ ]:





# In[2]:


starting_time = datetime.now()


# In[3]:


config_file = "config_batch"
address_file = "./address.csv.gz"
sample_size = None

import sys, getopt
opts, args = getopt.getopt(sys.argv[1:],"f:c:a:s:vq",[])

for opt, arg in opts:
    if opt == "-c":
        config_file = arg
        
    if opt == "-a":
        address_file = arg
        
    if opt == "-f":
        print("Run in jupyter ...", arg)
        AddressCleanserUtils.within_jupyter=True
        
    if opt == "-s":
        sample_size = int(arg)
        
    if opt == "-v": # Verbose
        logger.setLevel(logging.DEBUG)
        
    if opt == "-q": # quiet
        logger.setLevel(logging.WARNING)
        


# In[4]:


if AddressCleanserUtils.within_jupyter :
    log("Running in Jupyter, using hardcoded parameters")
#     config_file = "config_best"
#     address_file = "./best.csv.gz"
    config_file = "config_batch"
    address_file = "./address.csv.gz"

    sample_size = 10000
    AddressCleanserUtils.photon_host = "172.26.0.1:2322"
    AddressCleanserUtils.libpostal_host = "172.26.0.1:8080"

# with_dask=False
# %matplotlib inline


# In[5]:


import importlib
log(f"Loading config file {config_file}")
config_module = importlib.import_module(config_file)


# In[6]:


# Check that some required variables are present in the configuration file

field_names = ["street_field","housenbr_field","city_field","postcode_field", "country_field", "addr_key_field"]
#other_var_names = ["photon_host","osm_host","libpostal_host", "regex_replacements"]
other_var_names = ["regex_replacements"]
for var_name in field_names  + other_var_names:
    assert var_name in dir(config_module), var_name + " not defined in config module " + config_file


# In[ ]:





# In[7]:


AddressCleanserUtils.street_field    = config_module.street_field
AddressCleanserUtils.housenbr_field  = config_module.housenbr_field
AddressCleanserUtils.city_field      = config_module.city_field
AddressCleanserUtils.postcode_field  = config_module.postcode_field
AddressCleanserUtils.country_field  = config_module.country_field

AddressCleanserUtils.addr_key_field  = config_module.addr_key_field

AddressCleanserUtils.regex_replacements = config_module.regex_replacements

AddressCleanserUtils.use_osm_parent      = use_osm_parent
AddressCleanserUtils.with_rest_libpostal = with_rest_libpostal


# In[8]:


AddressCleanserUtils.pbar.register()


# In[9]:


# Check that Nominatim server is running properly
try: 
    osm = get_osm("Bruxelles")
    assert osm[0]["namedetails"]["name:fr"] == "Bruxelles"
    
    vlog("OSM working properly")
    
    
except Exception as e: 
    print("OSM not up & running")
    print("OSM host: ", AddressCleanserUtils.osm_host)
    raise e


# In[10]:


# In old version of Nominatim, page "details.php" could NOT return a JSON result, allowing to get place details from a place id
# In newer version, this has been added, allowing to get details about the parent of a place
# Is case "use_osm_parent" is true, check that "details.php" works correctly
if AddressCleanserUtils.use_osm_parent:
    try : 
        osm_det = get_osm_details(osm[0]["place_id"])
        assert osm_det["place_id"] == osm[0]["place_id"]

        vlog("OSM details working properly")

    except Exception as e: 
        print("OSM details not working")
        print("OSM host: ", AddressCleanserUtils.osm_host)
        raise e


# In[ ]:





# In[11]:


# Check that Photon server is running properly
try: 
    ph = get_photon("Bruxelles")
    assert ph["features"][0]["properties"]["name"] == "Brussels" 
    vlog("Photon working properly")
    
    
except Exception as e: 
    print("Photon not up & running ; Start it with 'nohup java -jar photon-*.jar &'")
    print("Photon host: ", AddressCleanserUtils.photon_host)
    raise e


# In[ ]:





# In[12]:


# Check that Libpostal is running properly
try: 
    lpost = parse_address("Bruxelles")
    assert lpost[0][0] == "bruxelles"
    vlog("Libpostal working properly")
except Exception as e: 
    print("Libpostal not up & running ")
    print("Libpostal: ", AddressCleanserUtils.libpostal_host)
    raise e


# # Data preparation

# In[13]:


# Get the addresses dataframe. Config module has to contain a "get_addresses(filename)" function, returning a dataframe, with 
# column names defined by variables (defined in config module) : street_field, housenbr_field, city_field, postcode_field , addr_key_field

log("Getting addresses")
addresses = config_module.get_addresses(address_file)
log(f"Got {addresses.shape[0]} addresses")
log(addresses)


# In[14]:


if sample_size and sample_size < addresses.shape[0]:
    log(f"Keep a sample of size {sample_size}")
    addresses = addresses.sample(sample_size)


# In[15]:


# Check that all required fields are present in addresses dataframe

for field in field_names:
    assert config_module.__getattribute__(field) in addresses, f"Field {field} missing in data !"


# In[16]:


# Check that the address identifier defined in config_module.addr_key_field is unique
assert addresses[addresses[config_module.addr_key_field].duplicated()].shape[0] == 0, "Key should be unique"


# In[17]:


vlog("Stripping and upper casing...")
addresses = addresses.apply(lambda col: col.fillna("").astype(str).str.strip().str.upper() if col.dtype.kind=='O' else col.astype(str) )


# # Main loop 

# In[18]:


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


# In[19]:


def main_loop(chunk):
    """
    Method "main_loop" processes the full cleansing sequence on a chunk of addresses : 
    - Apply a sequence of transformers (possibly empty)
    - Sent the (transformed) addresses to Nominatim
    - Parse and Check the results
    - For the addresses with no (accepted) result, try the next sequence of transformers
    """
    log(f"Chunk size: {chunk.shape[0]}")
    
    vlog(chunk)
    osm_addresses        = pd.DataFrame()
    rejected_addresses   = pd.DataFrame()
    stats = []
    
    init_chunk_size = chunk.shape[0]
    for transformers in transformers_sequence:

        vlog("--------------------------")
        vlog(f"| Transformers : { ';'.join(transformers) }")
        vlog("--------------------------")

#         display(chunk)
        osm_results, rejected, step_stats = transform_and_process(chunk, transformers, config_module.addr_key_field, 
                                                                  config_module.street_field, config_module.housenbr_field, 
                                                                  config_module.city_field, config_module.postcode_field,
                                                                  config_module.country_field, 
                                                                 check_osm_results=check_osm_results)

        osm_addresses =      osm_addresses.append(osm_results, sort=False).drop_duplicates()
        rejected_addresses = rejected_addresses.append(rejected, sort=False).drop_duplicates()

        vlog("Results: ")
        vlog(osm_results.head())
        vlog(osm_results.shape)

        vlog(f"Match rate so far: {osm_addresses.shape[0] / init_chunk_size if init_chunk_size > 0 else '(empty chunk size)'}")
             
        stats.append(step_stats)
        vlog(step_stats)

        chunk  = chunk[~chunk[config_module.addr_key_field].isin(osm_results[config_module.addr_key_field])].copy()

        ts = AddressCleanserUtils.timestats
        tot = np.sum([ts[key] for key in ts])
        if tot.total_seconds()>0:
            for key in ts: 
                vlog(f"{key:12}: {ts[key]} ({100*ts[key]/tot:.3} %)")
             
        vlog("")
        vlog("")
        vlog("####################")
        vlog("")
        vlog("")
        
    log("Chunk results: ")
    log(osm_addresses)
    log(f"Chunk match rate: {(osm_addresses.shape[0] / init_chunk_size) if init_chunk_size > 0 else '(empty chunk size)'}")
    log(pd.DataFrame(stats))
    
    return osm_addresses, rejected_addresses, stats     


# In[ ]:





# In[20]:


# Compute the number of chunks
min_nb_chunks= 4

if addresses.shape[0] > max_chunk_size * min_nb_chunks:
    chunk_size = max_chunk_size
    
elif addresses.shape[0] < min_chunk_size * min_nb_chunks:
    chunk_size = min_chunk_size
else: 
    chunk_size = int(np.sqrt(max_chunk_size *min_chunk_size))
log(f"Chunk_size: {chunk_size}")


# Do the main processing, with dask or simply in sequential chunks. 
# 
# Processing a chunk may require at some point a huge amount of memory. A single chunk with a few millions addresses may result in memory error ; this is why we split the main addresses dataframe is smaller chunks.
# 

# In[21]:


stats = []


if with_dask : 
    
    from dask.diagnostics import Profiler, ResourceProfiler
    
    #AddressCleanserUtils.with_dask = False
    
    # Sorting : allow to increase the probability to have duplicates within a chunk
    dd_to_process = dd.from_pandas(addresses.sort_values([config_module.postcode_field, config_module.street_field]).reset_index(drop=True), 
                                   chunksize=chunk_size)

    dask_task = dd_to_process.map_partitions(main_loop)

    with Profiler() as prof, ResourceProfiler() as rprof : 
        res =  dask_task.compute(scheduler='processes')

    log("All chunks done, gather all results...")
    
    osm_addresses      = pd.concat([chunk_osm_addresses      for (chunk_osm_addresses, _, _)      in res], sort=False)
    rejected_addresses = pd.concat([chunk_rejected_addresses for (_, chunk_rejected_addresses, _) in res], sort=False)

    for (_, _, chunk_stats) in res: 
        stats.extend(chunk_stats)

    log(f"Global match rate: { osm_addresses.shape[0]/addresses.shape[0] } ")

else: 
    #AddressCleanserUtils.with_dask = True

    osm_addresses        = pd.DataFrame()
    rejected_addresses   = pd.DataFrame()

    chunks_boundaries =  range(chunk_size, addresses.shape[0] , chunk_size)

    for chunk in tqdm(np.array_split(addresses.sort_values([config_module.postcode_field, config_module.street_field]), chunks_boundaries)):


        chunk_osm_addresses, chunk_rejected_addresses, chunk_stats = main_loop(chunk)


        osm_addresses =      osm_addresses.append(chunk_osm_addresses, sort=False).drop_duplicates()
        rejected_addresses = rejected_addresses.append(chunk_rejected_addresses, sort=False).drop_duplicates()

        log(f"Global match rate so far: {osm_addresses.shape[0]/addresses.shape[0]}")
        stats.extend(chunk_stats)


# In[22]:


# inclusion_test("NEU", "NEUCHATEAU")


# In[23]:


addresses


# In[24]:


# get_osm("6840 NEUFCHÂTEAU")


# In[25]:


if with_dask:
    from dask.diagnostics import visualize
    from bokeh.io import output_notebook, output_file
    output_file("dask_stats.html")
    # output_notebook()
    visualize([prof, rprof])


# In[26]:


# osm_addresses.SIM_street_which.value_counts() /osm_addresses.shape[0] #.plot.pie()


# # Rejected addresses

# Give some statistics about rejected adresses. 
# "rejected_addresses" contains two types of rejected addresses : 
# - rejected_addresses["reject_reason"] == "mismatch" : by comparing field by field input address and output address, this addresses has been rejected
# - rejected_addresses["reject_reason"] == "tail" : when OSM returns several results, only one is kept in "osm_addresses", all the others are put in rejected_addresses
# 
# Note that an addresse may have been rejected at a specific step (for a giver sequence of transformer), but not at another one.
# "rejected_addresses" may then contain a lot of addresses for which a result has been accepted further on.
# 
# "rejected_addresses_final" contains the only addresses for which all results have been rejected.
# 

# In[27]:


rejected_addresses_final =  rejected_addresses[rejected_addresses["reject_reason"] == "mismatch"]

rejected_addresses_final =  rejected_addresses_final[~rejected_addresses_final[config_module.addr_key_field].isin(osm_addresses[config_module.addr_key_field])]

# Needed with check_with_transformed = True (but doesn't hurt if not)
rejected_addresses_final = rejected_addresses_final.drop([config_module.street_field,
                                                          config_module.housenbr_field,
                                                          config_module.postcode_field,
                                                          config_module.city_field,
                                                          config_module.country_field],
                                                          axis=1
                                                        )

# print(rejected_addresses.keys())
# print(osm_addresses.keys())
# print(rejected_addresses.keys() & osm_addresses.keys())

rejected_addresses_final  = rejected_addresses_final.merge(addresses).sort_values(["SIM_street", config_module.addr_key_field])[["method", 
    config_module.addr_key_field, "osm_addr_in", 
    config_module.street_field, config_module.housenbr_field, config_module.postcode_field, config_module.city_field,  config_module.country_field,  
    "addr_out_street", "addr_out_city", "addr_out_number", "addr_out_postcode", "addr_out_other", "SIM_street", "SIM_zip"]].drop_duplicates()
log("Rejected addresses: ")
log(rejected_addresses_final)


# In[28]:


log(f"Number of unique rejected addresses: {rejected_addresses_final[config_module.addr_key_field].nunique()}")


# In[29]:


log(f"Number of unique city-streets in rejected addresses: {rejected_addresses_final[[config_module.postcode_field, config_module.street_field]].drop_duplicates().shape[0]}")


# In[30]:


rejected_addresses_final[rejected_addresses_final.addr_out_street.isnull()]


# In[31]:


rejected_addresses_final[rejected_addresses_final.addr_out_street.notnull()]#.drop(["method"], axis=1).drop_duplicates()


# In[32]:


# Swap street - city
log("Rejected addresses, but where it might have a swap between street and city")
str_cmp= street_compare(rejected_addresses_final[config_module.street_field], rejected_addresses_final.addr_out_city)
x= rejected_addresses_final[(str_cmp>0.5) & (rejected_addresses_final.addr_out_street.isnull()) & (rejected_addresses_final.SIM_zip >= 0.1)].drop_duplicates(subset=config_module.addr_key_field)
log(x)
log(f"Number of unique addresses: {x[config_module.addr_key_field].nunique()}")


# In[33]:


# Other mismatches
rejected_addresses_final[(str_cmp<=0.5) | (rejected_addresses_final.addr_out_street.notnull()) | (rejected_addresses_final.SIM_zip < 0.1)].drop_duplicates(subset=config_module.addr_key_field)


# # No match

# In[34]:


log("Addresses with no match (but some matches where rejected)")
log(addresses[~addresses[config_module.addr_key_field].isin(osm_addresses[config_module.addr_key_field]) & addresses[config_module.addr_key_field].isin(rejected_addresses[config_module.addr_key_field])])


# In[35]:


rejected_addresses


# In[36]:


log("Addresses with no match at all")
no_match = addresses[~addresses[config_module.addr_key_field].isin(osm_addresses[config_module.addr_key_field]) & ~addresses[config_module.addr_key_field].isin(rejected_addresses[config_module.addr_key_field])]
log(no_match)


# In[37]:


log(f"Number of unique city-streets in no match addresses: {no_match[[config_module.postcode_field, config_module.street_field]].drop_duplicates().shape[0]}")


# In[38]:


log("Main cities in no match addresses: ")
log(no_match[config_module.city_field].value_counts().head(10))


# In[39]:


log("Main streets in no match addresses: ")
log(no_match[config_module.street_field].value_counts().head(10))


# # Extra house number

# In many situation, OSM does not return a correct house number : 
# - Either because the building is not known by OSM. In this case, house number is empty in result
# - Or because house number in input also contains information such as box, level...
# 
# We then consider that house number is not reliable enough and compute our own house number field, named "extra_house_nbr"

# In[40]:


log("Add extra house number")
osm_addresses = add_extra_house_number(osm_addresses, addresses, street_field=config_module.street_field, housenbr_field=config_module.housenbr_field)


# In[41]:


# osm_addresses.drop("extra_house_nbr", axis=1, inplace=True)


# In[42]:


ex_hs_nb = osm_addresses[[config_module.addr_key_field, "osm_addr_in", "extra_house_nbr", "addr_out_number"]].replace("", np.NaN)


# In[43]:


log("Add new information: ")
log(ex_hs_nb[ex_hs_nb.addr_out_number.isnull() & ex_hs_nb.extra_house_nbr.notnull()])


# In[44]:


log("No number at all: ")
log(ex_hs_nb[ex_hs_nb.addr_out_number.isnull() & ex_hs_nb.extra_house_nbr.isnull()])


# In[45]:


log("Agreed: ")
log(ex_hs_nb[ex_hs_nb.addr_out_number.notnull() & ex_hs_nb.extra_house_nbr.notnull() & (ex_hs_nb.addr_out_number == ex_hs_nb.extra_house_nbr)])


# In[46]:


log("Disagreed: ")
log(ex_hs_nb[ex_hs_nb.addr_out_number.notnull() & ex_hs_nb.extra_house_nbr.notnull() & (ex_hs_nb.addr_out_number != ex_hs_nb.extra_house_nbr)])


# In[47]:


log("Error: ") # There were no number in input, but OSM found one
log(ex_hs_nb[ex_hs_nb.addr_out_number.notnull() & ex_hs_nb.extra_house_nbr.isnull()])


# In[48]:


extra_address_stats = {
    "New information" : (ex_hs_nb.addr_out_number.isnull()  & ex_hs_nb.extra_house_nbr.notnull()).sum(),
    "No number at all": (ex_hs_nb.addr_out_number.isnull()  & ex_hs_nb.extra_house_nbr.isnull() ).sum(),
    "Agree"  :          (ex_hs_nb.addr_out_number.notnull() & ex_hs_nb.extra_house_nbr.notnull() & (ex_hs_nb.addr_out_number == ex_hs_nb.extra_house_nbr)).sum(),
    "Disagree":         (ex_hs_nb.addr_out_number.notnull() & ex_hs_nb.extra_house_nbr.notnull() & (ex_hs_nb.addr_out_number != ex_hs_nb.extra_house_nbr)).sum(),
    "Error" :           (ex_hs_nb.addr_out_number.notnull() & ex_hs_nb.extra_house_nbr.isnull()).sum()
    
}
extra_address_stats = pd.DataFrame(extra_address_stats, index=["Count"]).T
log(extra_address_stats)


# In[49]:


# extra_address_stats.Count.plot.pie(label="",  autopct='%1.1f%%')


# In[50]:


assert extra_address_stats.Count.sum() == osm_addresses.shape[0]


# # Some stats

# In[51]:


_stats = pd.DataFrame(stats)[["method","todo", "sent", "match", "match_26", "reject_rec", "reject_addr", "reject_mism"]]
_stats = _stats.reset_index().groupby("method").sum().reset_index().sort_values("index").drop("index", axis=1)


# In[52]:


assert osm_addresses.shape[0] == _stats["match"].sum()


# In[53]:


log(f"Global match rate : {osm_addresses.shape[0]/addresses.shape[0]}")


# In[54]:


rejected_count = rejected_addresses[~rejected_addresses[config_module.addr_key_field].isin(osm_addresses[config_module.addr_key_field])][config_module.addr_key_field].nunique()
rejected_count

nomatch_count = addresses[~addresses[config_module.addr_key_field].isin(osm_addresses[config_module.addr_key_field]) & ~addresses[config_module.addr_key_field].isin(rejected_addresses[config_module.addr_key_field])].shape[0]

rejected_count, nomatch_count


# In[55]:


#rejected_addresses[~rejected_addresses[config_module.addr_key_field].isin(osm_addresses[config_module.addr_key_field])]


# In[56]:


# osm_addresses[osm_addresses.EntityNumber == "2.227.707.047"]


# In[57]:


missing_address_count = addresses.shape[0] - osm_addresses.shape[0]

assert rejected_count + nomatch_count == missing_address_count

# print("Missing : ", missing_address_count)


# In[58]:


_stats = _stats.append(pd.DataFrame([{"method": "reject", "todo": rejected_count, "match": rejected_count},
                              {"method": "nomatch", "todo": nomatch_count, "match": nomatch_count},
                             ]), sort=False)


# In[59]:


_stats["match rate"] = _stats["match"]/_stats["sent"]
_stats["glob match rate"] = _stats["match"]/addresses.shape[0]

log(_stats[_stats.match > 0])#.sort_values("match", ascending=False)


# In[60]:


#


# In[61]:


if AddressCleanserUtils.within_jupyter:
    import matplotlib.pyplot as plt

    _stats.set_index("method").match.plot.pie()
    plt.tight_layout()


# In[62]:


log(f"Place ranks: \n{osm_addresses.place_rank.value_counts().to_string()}")


# In[63]:


osm_addresses.place_rank.value_counts() / osm_addresses.shape[0]


# In[64]:


if AddressCleanserUtils.within_jupyter:
    osm_addresses.place_rank.value_counts().plot.bar()


# In[65]:


if AddressCleanserUtils.within_jupyter:
    osm_addresses.place_rank.value_counts().plot.pie()


# In[66]:


if AddressCleanserUtils.within_jupyter:
    osm_addresses.addr_out_number.isnull().value_counts().plot.bar()


# In[67]:


if AddressCleanserUtils.within_jupyter:
    addresses[config_module.housenbr_field].isnull().value_counts().plot.bar()


# In[68]:


# Remark : only works when dask is not used 
# Gives times used of transformer, querying & processing osm, and checking results
if not with_dask:
    ts = AddressCleanserUtils.timestats
    tot = np.sum([ts[key] for key in ts])
    for key in ts: 
        log(f"{key:12}: {ts[key]} ({100*ts[key]/tot:.3} %)")


# In[85]:


log("Country statistics")
x = addresses.merge(osm_addresses, how="outer") #[[config_module.country_field, "addr_out_country"]].value_counts()
log(pd.crosstab(x[config_module.country_field].fillna("[none]"), x["addr_out_country"].fillna("[none]"), margins=True))


# # Output

# In[ ]:


output_folder = address_file.rsplit(".", 1)[0]

import os

try:
    os.mkdir(output_folder)
except OSError:
    log ("Creation of the directory %s failed" % output_folder)
else:
    log ("Successfully created the directory %s " % output_folder)
    
    
output_filename_xls = output_folder + "/match.xlsx"
output_filename_pkl = output_folder + "/match.pkl"

nomatch_filename = output_folder + "/nomatch.xlsx"

reject_filename = output_folder + "/reject.xlsx"

stats_filename = output_folder + "/stats.xlsx"


# In[ ]:


final_output = addresses.merge(osm_addresses, how="left")


log(f"Writing results on {output_filename_xls} ...")
try: 
    final_output.to_excel(output_filename_xls)
except Exception as e: 
    log("Failed ! ")
    log(e)
    
log(f"Writing results on {output_filename_pkl} ...")
try: 
    final_output.to_pickle(output_filename_pkl)
except Exception as e: 
    log("Failed ! ")
    log(e)
    


# In[ ]:





# In[ ]:


log(f"Writing rejected on {reject_filename} ...")
try: 

    rejected_addresses_final.sort_values([config_module.addr_key_field]).set_index([config_module.addr_key_field,
                                                                               config_module.street_field,
                                                                               config_module.housenbr_field,
                                                                               config_module.postcode_field,
                                                                               config_module.city_field,
                                                                               config_module.country_field,
                                                                               "method"]).to_excel(reject_filename)
except Exception as e: 
    log("Failed ! ")
    log(e)
    

log(f"Writing nomatch on {nomatch_filename} ...")
try: 
    nomatch =  addresses[~addresses[config_module.addr_key_field].isin(osm_addresses[config_module.addr_key_field]) & ~addresses[config_module.addr_key_field].isin(rejected_addresses[config_module.addr_key_field])]
    nomatch.to_excel(nomatch_filename)
except Exception as e: 
    log("Failed ! ")
    log(e)
    


# In[ ]:


log(f"Writing stats on {stats_filename} ...")
try: 
    with pd.ExcelWriter(stats_filename) as writer:
        _stats.to_excel(writer, "match_rate")
        
        pr_vc = osm_addresses.place_rank.value_counts()
        pr_vc = pd.concat([pr_vc, pr_vc/ osm_addresses.shape[0]], axis=1)
        pr_vc.columns = ["Count", "%"]
        pr_vc.to_excel(writer, "place_rank")

except Exception as e: 
    log("Failed ! ")
    log(e)
    


# In[ ]:


log("Done !")
log(f"Total time : {datetime.now() - starting_time}")

