import argparse
import csv
import json
import gzip
import calendar
import datetime
from geocoding import Geocoder
    
city_mapping = {
    "Cap" : "Cap-Haitien",
    "S. Marc" : "Saint-Marc",
    "" : "unknown",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--location_cache", dest="location_cache", help="")    
    args, rest = parser.parse_known_args()

    geocoder = Geocoder(args.location_cache)
    
    products = {}
    listings = {}
    cities = {}
    for fname in args.inputs:
        with gzip.open(fname, "rt") as ifd:
            c = csv.DictReader(ifd, delimiter="\t")
            for i, row in enumerate(c):
                product_id = "{}_{}".format(row["product_name"], row["listing_product_description"])
                products[product_id] = {"product_name" : row["product_name"],
                                        "product_description" : row["listing_product_description"],
                                        "entity_type" : "product",
                }
                row["city_name"] = city_mapping.get(row["city_name"], row["city_name"])
                city_id = "city_{}".format(row["city_name"])
                if city_id not in cities:                    
                    cities[city_id] = {"city_name" : row["city_name"], "entity_type" : "city"}
                    if row["city_name"] != "unknown":
                        g = None
                        while g == None:
                            try:
                                g = geocoder("{}".format(row["city_name"]))
                            except:
                                g = None
                                print(row["city_name"])
                        cities[city_id]["city_coordinates"] = g #{"latitude" : g.latitude, "longitude" : g.longitude}
                    
                listing_id = "listing_{}".format(i)
                listings[listing_id] = {"availability" : row["listing_availability"],                    
                                        "qualifier" : row["listing_qualifier"],
                                        "price_for" : product_id,
                                        "price_in" : city_id,
                                        "entity_type" : "listing",
                }
                try:
                    listings[listing_id]["price_high"] = float(row["listing_price_high"])
                except:
                    pass
                try:
                    listings[listing_id]["price_low"] = float(row["listing_price_low"])
                except:
                    pass
                listings[listing_id]["date"] = row["listing_date"]
    with gzip.open(args.output, "wt") as ofd:
        for eid, entity in products.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")
        for eid, entity in cities.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")
        for eid, entity in listings.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")            
