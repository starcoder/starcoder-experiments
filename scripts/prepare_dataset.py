import argparse
import gzip
import pickle
import logging
import json
from typing import IO, Any
from starcoder.dataset import Dataset
from starcoder.schema import Schema
from starcoder import field
from starcoder.registry import field_classes

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_input", dest="schema_input", help="Schema JSON input file")
    parser.add_argument("--data_input", dest="data_input", help="Data JSON input file")
    parser.add_argument("--schema_output", dest="schema_output", help="Schema pickle output file")
    parser.add_argument("--data_output", dest="data_output", help="Data pickle output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.schema_input, "rt") as ifd:
        schema = Schema(json.load(ifd), field_classes)
    
    entities = []
    with gzip.open(args.data_input, "rt") as ifd: # type: ignore
        for entity in map(json.loads, ifd):
            if entity[schema.entity_type_field.name] in schema.entity_types:
                schema.observe_entity(entity)
                entities.append(entity)
    schema.minimize()
    schema.verify()
    logging.info("Created %s", schema)

    dataset = Dataset(schema, entities, strict=False)
    
    with gzip.open(args.schema_output, "wb") as ofd:
        pickle.dump(schema, ofd)    
    with gzip.open(args.data_output, "wb") as ofd:
        pickle.dump(dataset, ofd)
    logging.info("Created %s from %s and %s, wrote to %s and %s",
                 dataset,
                 args.data_input,
                 args.schema_input,
                 args.schema_output,
                 args.data_output)
