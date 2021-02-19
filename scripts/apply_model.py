import sys
import gzip
import pickle
import logging
import argparse
import torch
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
    parser.add_argument("--batch_size", dest="batch_size", default=1512, type=int, help="")
    parser.add_argument("--log_level", dest="log_level", default="ERROR", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], help="Logging level")
    args, rest = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )
    
    
    with gzip.open(args.model, "rb") as ifd:
        model = torch.load(ifd) # type: ignore
        #state, margs, _ = torch.load(ifd) # type: ignore
    #with (gzip.open if args.schema.endswith("gz") else open)(args.schema, "rt") as ifd:
    #    schema = json.loads(ifd.read())


    data = []
    with (gzip.open if args.dataset.endswith("gz") else open)(args.dataset, "rt") as ifd:
        for line in ifd:
            entity = json.loads(line)
            eid = entity[model.schema["id_property"]]
            data.append(entity)
    data = Dataset(model.schema, data)
    
        
    #logger.info("Loading model")
    #model = GraphAutoencoder(schema, 
    #                         data=data)
    #margs.depth, 
    #                         margs.autoencoder_shapes,
    #                         reverse_relationships=True,
    #                         depthwise_boost=margs.depthwise_boost
    #)
    #model.load_state_dict(state)
    model.eval()
    if args.gpu:
        model.cuda()
        logging.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                     torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)
        
    model.train(False)
    if args.split == None:
        components = [i for i in range(data.num_components)]
    else:
        data = data.subselect_entities(entity_ids)
        components = [i for i in range(data.num_components)]
    
    logging.info("Dataset has %d entities", data.num_entities)
    all_bottlenecks = {}
    all_reconstructions = {}
    batchifier = starport(model.schema["meta"]["batchifier"])(model.schema)
    
    for originals, loss, reconstructions, bottlenecks in apply_to_components(
            model,
            batchifier, #batchifier_classes["sample_components"]([]),
            data,
            args.batch_size,
            args.gpu,
            args.mask_properties,
            args.mask_probability
    ):

        #for entity in [data.schema.unpack(e) for e in unstack_entities(norm, data.schema)]:
        for rid, reconstruction in reconstructions.items(): #unstack_entities(norm, data.schema):
            all_reconstructions[rid] = reconstruction
            all_bottlenecks[rid] = bottlenecks[rid]
            #print(entity)
            #entity = {k : model.property_objects[k].unpack(v) if k not in [data.schema["id_property"], data.schema["entity_type_property"]] else v for k, v in entity.items()}
            #reconstructions[rec] = reconstruction
        #    bottlenecks[entity[data.schema["id_property"]]] = bns[entity[data.schema["id_property"]]].tolist()
            
    with gzip.open(args.output, "wt") as ofd:
        for eid, reconstruction in reconstructions.items():
            item = {"original" : data.entity(eid),
                    "reconstruction" : all_reconstructions[eid],
                    "bottleneck" : all_bottlenecks[eid]
            }
            ofd.write(json.dumps(item) + "\n")
    logger.info("Saved reconstructions to '%s'", args.output)
