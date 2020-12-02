import sys
import gzip
import pickle
import logging
import argparse
import torch
from starcoder.registry import field_model_classes, batchifier_classes, field_classes, scheduler_classes
from starcoder.ensemble_model import GraphAutoencoder
from starcoder.dataset import Dataset
from starcoder.schema import Schema
from starcoder.field import CategoricalField
from starcoder.entity import stack_entities, UnpackedEntity, PackedEntity, Index, ID, unstack_entities
from starcoder.adjacency import Adjacencies
from starcoder.utils import apply_to_components
import json
import tempfile
import random
import os
from typing import List, Type

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", help="Input dataset file")
    parser.add_argument("--split", dest="split", default=None, help="Data split (if none specified, run over everything)")
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--mask_field", dest="mask_field", help="Mask field")
    parser.add_argument("--mask_probability", dest="mask_probability", default=0.5, help="Mask probability")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--gpu", dest="gpu", default=False, action="store_true", help="Use GPU")
    parser.add_argument("--batch_size", dest="batch_size", default=512, type=int, help="")
    parser.add_argument("--log_level", dest="log_level", default="INFO", 
                        choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args, rest = parser.parse_known_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))

    with gzip.open(args.model, "rb") as ifd:
        state, margs, schema = torch.load(ifd) # type: ignore

    with gzip.open(args.dataset, "rb") as ifd:
        dataset = pickle.load(ifd) # type: ignore
    
    model = GraphAutoencoder(schema, 
                             margs.depth, 
                             margs.autoencoder_shapes,
                             reverse_relationships=True,
    )
    model.load_state_dict(state)
    model.eval()
    
    if args.gpu:
        model.cuda()
        logging.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                     torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)

    model.train(False)
    if args.split == None:
        components = [i for i in range(dataset.num_components)]
    else:
        raise Exception()
        #with gzip.open(args.split, "rb") as ifd:
        #    indices = pickle.load(ifd) # type: ignore
        #    dataset = dataset.subselect_entities_by_index(indices)
    
    logging.info("Dataset has %d entities", dataset.num_entities)
    bottlenecks = {}
    reconstructions = {}
    for decoded_fields, norm, bns, masking in apply_to_components(model, batchifier_classes["sample_components"]([]), dataset, args.batch_size, args.gpu, args.mask_field, args.mask_probability):
        #print({k : type(v) for k, v in norm.items()})
        #print({k : type(v) for k, v in decoded_fields.items()})
        #print(norm["street_coordinates"].shape)
        #print(decoded_fields["street_coordinates"].shape)
        #first = list(norm["id"])
        #second = list(bns.keys())
        #print([x for x in first if x not in second])
        #print([x for x in second if x not in first])
        for entity in [dataset.schema.unpack(e) for e in unstack_entities(norm, dataset.schema)]:
            reconstructions[entity[dataset.schema.id_field.name]] = entity
            bottlenecks[entity[dataset.schema.id_field.name]] = bns[entity[dataset.schema.id_field.name]].tolist()
            
    with gzip.open(args.output, "wt") as ofd:
        for eid, reconstruction in reconstructions.items():
            item = {"original" : dataset.entity(eid),
                    "reconstruction" : reconstructions[eid],
                    "bottleneck" : bottlenecks[eid]
            }
            ofd.write(json.dumps(item) + "\n")
    logger.info("Saved reconstructions to '%s'", args.output)
