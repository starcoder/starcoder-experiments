import sys
import gzip
import pickle
import logging
import argparse
import torch
from torch import Tensor
from starcoder.ensemble_model import GraphAutoencoder
from starcoder.dataset import Dataset
from starcoder.property import CategoricalProperty
from starcoder.utils import apply_to_components, starport
import json
import tempfile
import random
import os

logger = logging.getLogger("apply_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", dest="schema", help="Schema file")
    parser.add_argument("--dataset", dest="dataset", help="Input dataset file")
    parser.add_argument("--split", dest="split", default=None, help="Data split (if none specified, run over everything)")
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--mask_properties", dest="mask_properties", default=[], nargs="+", help="Mask properties")
    parser.add_argument("--mask_probability", dest="mask_probability", default=0.5, help="Mask probability")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--gpu", dest="gpu", default=False, action="store_true", help="Use GPU")
    parser.add_argument("--blind", dest="blind", default=False, action="store_true", help="Blind")
    parser.add_argument("--remove_structure", dest="remove_structure", default=False, action="store_true", help="Remove structure")
    parser.add_argument("--batch_size", dest="batch_size", default=1512, type=int, help="")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], help="Logging level")
    args, rest = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )

    with open(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())
    if "shared_entity_types" in schema["meta"]:
        del schema["meta"]["shared_entity_types"]

    
    with gzip.open(args.model, "rb") as ifd:
        model = torch.load(ifd)

    model.eval()
    if args.gpu:
        model.cuda()
        logging.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                     torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)

    model.train(False)

    data = []
    with (gzip.open if args.dataset.endswith("gz") else open)(args.dataset, "rt") as ifd:
        for line in ifd:
            entity = json.loads(line)
            eid = entity[model.schema["id_property"]]
            if args.remove_structure == True:
                entity = {k : v for k, v in entity.items() if k not in schema["relationships"]}
            data.append(entity)

    dataset = Dataset(schema, data)

    #if args.split == None:
    #    components = [i for i in range(dataset.num_components)]
    #else:
    #    dataset = dataset.subselect_entities(entity_ids)
    #    components = [i for i in range(dataset.num_components)]

    logging.info("Dataset has %d entities", dataset.num_entities)
    all_bottlenecks = {}
    all_reconstructions = {}
    #batchifier = starport(schema["meta"]["batchifier"])(schema)
    batchifier = starport("starcoder.batchifier.SampleEntities")(schema)

    logging.info("Running full reconstruction")
    for originals, loss, reconstructions, bottlenecks in apply_to_components(
            model,
            batchifier,
            dataset,
            dataset.num_entities,
            #args.batch_size,
            args.gpu,
            args.mask_properties,
            args.mask_probability
    ):
        for rid, reconstruction in reconstructions.items():
            all_reconstructions[rid] = {k : v.tolist() if isinstance(v, Tensor) else {kk : vv.tolist() if isinstance(vv, Tensor) else vv for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in reconstruction.items()}
            all_bottlenecks[rid] = bottlenecks[rid]
            
    if args.blind:
        for property_name in schema["properties"].keys():
            logging.info("Running blind reconstruction for property '%s'", property_name)
            blind_dataset = Dataset(schema, [{k : v for k, v in d.items() if k != args.blind} for d in data])
            if args.split == None:
                components = [i for i in range(blind_dataset.num_components)]
            else:
                blind_dataset = blind_dataset.subselect_entities(entity_ids)
                components = [i for i in range(blind_dataset.num_components)]
            for originals, loss, reconstructions, bottlenecks in apply_to_components(
                    model,
                    batchifier,
                    blind_dataset,
                    dataset.num_entities,
                    #args.batch_size,
                    args.gpu,
                    args.mask_properties,
                    args.mask_probability
            ):
                for rid, reconstruction in reconstructions.items():
                    if property_name in reconstruction:
                        v = reconstruction[property_name]
                        all_reconstructions[rid][property_name] = v.tolist() if isinstance(v, Tensor) else {kk : vv.tolist() if isinstance(vv, Tensor) else vv for kk, vv in v.items()} if isinstance(v, dict) else v

    with gzip.open(args.output, "wt") as ofd:
        for eid, reconstruction in reconstructions.items():
            item = {"original" : dataset.entity(eid),
                    "reconstruction" : all_reconstructions[eid],
                    "bottleneck" : all_bottlenecks[eid]
            }
            ofd.write(json.dumps(item) + "\n")
    logger.info("Saved reconstructions to '%s'", args.output)
