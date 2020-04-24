# Config for file coming from KBO open data : https://kbopub.economie.fgov.be/kbo-open-data/login?lang=en

import pandas as pd
import numpy as np

# path = "./"

# addresses_filename  = path + "address.csv.gz" 

street_field   = "StreetFR"
housenbr_field = "HouseNumber"
city_field     = "MunicipalityFR"
postcode_field = "Zipcode"
country_field = "CountryFR"
#addr_key_field = "addr_index"
addr_key_field = "EntityNumber"

sample_size = None# 1000

regex_replacements =  {
    "init": [
        [street_field,  "\(.*\)$", ""],
        [street_field,  "\' ", "'"],
        [street_field,  "(?i)[, ]*[SZ]N$", ""],
        [street_field,  " [ ]+", " "],


        [postcode_field,  "\.0$", ""],
        [housenbr_field,  "^-$", ""],
        [housenbr_field,  "^(?i)[\.szn]+$", ""],
    ],
 
     "lpost":  [
        # Keep only numbers
        [housenbr_field, "^([0-9]*)(.*)$", "\g<1>"],

        # Av --> avenue, chée...

        [street_field, "^r[\. ]",  "rue "],
        [street_field, "^av[\. ]", "avenue "],
        [street_field, "^ch([ée]e)?[\. ]",   "chaussée "],
        [street_field, "^b[lvd]{0,3}[\. ]",  "boulevard "],

        # rue d anvers --> rue d'anvers
        [street_field, "(avenue|rue|chauss[ée]e|boulevard) d ", "\g<1> d'"],
        [street_field, "(avenue|rue|chauss[ée]e|boulevard) de l ", "\g<1> de l'"],

        [street_field, " de l ", " de l'"],

        # St Gilles --> Saint Gilles
        [city_field, "st (jans|joost|lambrechts|pieters|giillis|niklaas|truiden)" , "sint \g<1>"],
        [city_field, "st (josse|jean|gilles|lambert|pierre|etienne|nicolas)" , "saint \g<1>"],
        [city_field, "ste? agathe",  "sainte agathe"],
    
    ]
}




def get_addresses(addresses_filename):
    
    addresses = pd.read_csv(addresses_filename,  
                            usecols = [addr_key_field,
                                       country_field, 
                                       postcode_field, 
                                       city_field, 
                                       street_field, 
                                       housenbr_field])

    #addresses = addresses.rename(columns={"index":addr_key_field})
    
    addresses = addresses[addresses[street_field].notnull() & addresses[city_field].notnull() & addresses[country_field].isnull() ]
    
    addresses[postcode_field] = addresses[postcode_field].astype(str)
    
    if sample_size: 
        addresses = addresses.sample(sample_size, random_state=0)
    
    return addresses.drop(country_field, axis=1)
