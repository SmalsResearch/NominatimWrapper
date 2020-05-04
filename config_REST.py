use_osm_parent = True

with_timing_info = False

street_field  = "street"
housenbr_field = "housenumber"
postcode_field = "postcode"
city_field  =    "city"
country_field =  "country"
addr_key_field = "addr_key"

regex_replacements = {
    "init": [
        [street_field, "^(.+)\(((AV[E .]|CH[A .]|RUE|BOU|B[LVD]+|PL[A .]|SQ|ALL|GAL)[^\)]*)\)$", "\g<2> \g<1>"],
        [street_field, "[, ]*(SN|ZN)$", ""],
        [street_field, "' ", "'"],
        [street_field,  "\(.+\)$", ""],
    ],
    "lpost": [
        # Keep only numbers
        [housenbr_field, "^([0-9]*)(.*)$", "\g<1>"],

        # Av --> avenue, chée...

        [street_field, "^r[\. ]",  "rue "],
        [street_field, "^av[\. ]", "avenue "],
        [street_field, "^ch([ée]e)?[\. ]", "chaussée "],
        [street_field, "^b[lvd]{0,3}[\. ]",     "boulevard "],

        # rue d anvers --> rue d'anvers
        [street_field, "(avenue|rue|chauss[ée]e|boulevard) d ", "\g<1> d'"],
        [street_field, "(avenue|rue|chauss[ée]e|boulevard) de l ", "\g<1> de l'"],

        [street_field, " de l ", " de l'"]
    ]
}

    
