import argparse
import csv
import json
import gzip

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

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
                city_id = row["city_name"]
                cities[city_id] = {"city_name" : row["city_name"], "entity_type" : "city"}
                listing_id = "listing_{}".format(i)
                listings[listing_id] = {"date" : 0,
                                        "price_low" : row["listing_price_low"],
                                        "price_high" : row["listing_price_high"],
                                        "availability" : row["listing_availability"],
                                        "qualifier" : row["listing_qualifier"],
                                        "price_for" : product_id,
                                        "located_in" : city_id,
                                        "entity_type" : "listing",
                                        }

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
