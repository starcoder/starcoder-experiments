import argparse
import json
from jsonpath_ng.ext import parse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="configs", nargs="+", help="Configuration files")
    parser.add_argument("--schema", dest="schema", help="Schema file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    with open(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())
    for spec_type in ["properties", "entity_types", "relationships"]:
        schema[spec_type] = [{k : v for k, v in list(vals.items()) + [("name", name)]} for name, vals in schema.get(spec_type, {}).items()]


    for fname in args.configs:
        with open(fname, "rt") as ifd:
            for pattern_string, values in json.loads(ifd.read()):
                pattern = parse(pattern_string)
                for match in pattern.find(schema):
                    match.value["meta"] = match.value.get("meta", {})
                    for k, v in values.items():
                        match.value["meta"][k] = v

    for spec_type in ["properties", "entity_types", "relationships"]:
        schema[spec_type] = {vals["name"] : {k : v for k, v in vals.items() if k != "name"} for vals in schema.get(spec_type, [])}

    with open(args.output, "wt") as ofd:
        ofd.write(json.dumps(schema, indent=2))
