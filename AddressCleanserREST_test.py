#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import urllib

import numpy as np

import json

from tqdm.autonotebook import  tqdm

#%matplotlib inline

tqdm.pandas(tqdm)

import dask.dataframe as dd

from dask.multiprocessing import get
from dask.diagnostics import ProgressBar

from datetime import datetime
import matplotlib.pyplot as plt

from IPython.display import display


# In[50]:


from config_KBO import * 


# In[58]:


def call_ws(addr_data): #lg = "en,fr,nl"
    t = datetime.now()
    
    params = urllib.parse.urlencode({"street": addr_data[street_field],
                                     "housenumber": addr_data[housenbr_field],
                                     "city": addr_data[city_field],
                                     "postcode": addr_data[postcode_field]
                                    })
    url = "http://172.17.0.3:5000/search/?%s"%params
    
    
    try:
        with urllib.request.urlopen(url) as response:
            res = response.read()
            res = json.loads(res)
#             print(res)
            res["time"] = datetime.now() - t
            return res
    except Exception as e:
        return str(e)
    


# In[158]:


def get_plots(addresses, column):
    fig, axes = plt.subplots(ncols=1, nrows = addresses[column].nunique(),  sharex=True, figsize=[6, 3*addresses[column].nunique()])

    addresses.hist(column="_time", by=column,ax=axes)

    return fig


# In[51]:


addresses = get_addresses()


# In[98]:


display(addresses)


# In[100]:


with_dask = False
if with_dask : 
    dd_addresses = dd.from_pandas(addresses, npartitions=4)

    dask_task = dd_addresses.apply(call_ws, meta=('x', 'str'), axis=1)

    with ProgressBar(): 
        addresses["json"] = dask_task.compute()
else: 
    
    addresses["json"] = addresses.progress_apply(call_ws, axis=1)


# In[101]:


# Flask : 1:03
# gunicorn -w 1: 0:55
# gunicorn -w 2, npartition=2: 0:30
# gunicorn -w 2, npartition=4: 0:30
# gunicorn -w 4, npartition=4: 0:33

addresses


# In[102]:


addresses["status"]= addresses.json.apply(lambda d: "error" if "error" in d else "match" if "match" in d else "rejected")
addresses["time"]  = addresses.json.apply(lambda d: d["time"])

addresses["timing"]  = addresses.json.apply(lambda d: d["timing" if "timing" in d else {}])

addresses["method"]= addresses.json.apply(lambda d: d["match"][0]["method"] if len(d)>0 and "match" in d else "none")
addresses["street"]= addresses.json.apply(lambda d: d["match"][0]["addr_out_street"] if len(d)>0 and "match" in d else "")


display(addresses.drop("json", axis=1))


# In[ ]:


addresses["timing"].apply(pd.Series)


# In[103]:


display(addresses.status.value_counts())


# In[104]:


display(addresses.method.value_counts())


# In[163]:


addresses["_time"] = addresses.time.apply(lambda t: t.total_seconds())


# In[ ]:


print("Method : mean")


# In[166]:


display(addresses.groupby("method")._time.mean())


# In[ ]:


print("Method : std")


# In[171]:


display(addresses.groupby("method")._time.std())


# In[170]:





# In[172]:


print("Status : mean")


# In[173]:


display(addresses.groupby("status")._time.mean())


# In[174]:


print("Status : std")


# In[175]:


display(addresses.groupby("status")._time.std())


# In[159]:


fig = get_plots(addresses, "status")


# In[160]:


fig.savefig("time_per_status.png")


# In[161]:


fig = get_plots(addresses, "method")


# In[162]:


fig.savefig("time_per_method.png")


# In[ ]:




