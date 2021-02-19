import tracemalloc
import pickle
import re
import gzip
import sys
import argparse
import json
import logging
import warnings
import numpy
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer
import torch.profiler
import torch
from torch import Tensor
#from starcoder.registry import property_model_classes, batchifier_classes, property_classes, scheduler_classes
from starcoder.ensemble_model import GraphAutoencoder
from starcoder.dataset import Dataset
#from starcoder.schema import Schema
from starcoder.batchifier import Batchifier
from starcoder.property import CategoricalProperty
from starcoder.random import random
from starcoder.utils import run_epoch, simple_loss_policy, starport
from torch.autograd import set_detect_anomaly
set_detect_anomaly(True)
from typing import Dict, List, Any, Tuple
import os
import torch.autograd

logger = logging.getLogger("train_model")

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input-related
    parser.add_argument("--schema", dest="schema", help="Schema file")
    parser.add_argument("--data", dest="data", help="Input data file")
    parser.add_argument("--train", dest="train", help="Train split components file")
    parser.add_argument("--dev", dest="dev", help="Dev split components file")
    parser.add_argument("--gpu", dest="gpu", default=False, action="store_true", help="Use GPU")

    # output-related
    parser.add_argument("--model_output", dest="model_output", help="Model output file")

    # miscellaneous
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int, help="Random seed")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], help="Logging level")
    parser.add_argument("--profile", dest="profile", default=False, action="store_true")
    args, rest = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.random_seed != None:
        logger.info("Setting random seed to %d across the board", args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        numpy.random.seed(args.random_seed)

    with (gzip.open if args.schema.endswith("gz") else open)(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())



    with (gzip.open if args.train.endswith("gz") else open)(args.train, "rt") as ifd:
        train_ids = set(json.loads(ifd.read()))

    with (gzip.open if args.dev.endswith("gz") else open)(args.dev, "rt") as ifd:
        dev_ids = set(json.loads(ifd.read()))

    train, dev = [], []
    with (gzip.open if args.data.endswith("gz") else open)(args.data, "rt") as ifd:
        for line in ifd:            
            entity = json.loads(line)
            eid = entity[schema["id_property"]]
            if eid in train_ids:
                train.append(entity)
            # entities may occur in both train and dev data
            if eid in dev_ids:
                dev.append(entity)
    train = Dataset(schema, train)
    dev = Dataset(schema, dev)
    logger.info("Loaded %d train and %d dev entities", len(train), len(dev))

    batchifier = starport(schema["meta"]["batchifier"])(schema)

    #global_best_dev_loss = torch.tensor(numpy.nan)
    #global_best_state = None
    #best_trace: List[Dict[str, Any]]
    #for restart in [1]: #range(args.random_restarts + 1):

    local_best_dev_loss = torch.tensor(numpy.nan)
    local_best_state: Dict[str, Tensor]
    current_trace = []

    model = GraphAutoencoder(schema=schema,
                             data=train,
    ) 
    if args.gpu:
        model.cuda()
        logger.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                     torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)

    logger.info("Model: %s", model)
    logger.info("Model has %d parameters", model.parameter_count)

    model.init_weights()
    optim: Optimizer
    logger.info("Using Adam")        
    optim = Adam(model.parameters(), lr=0.001)
    patience = 10
    early_stop = 20
    sched = starport(schema["meta"]["scheduler"])(early_stop, optim, patience=patience, verbose=True)
    best_dev_loss = None
    best_state = None

    #logger.info("Warming up property encoders")
    #warmup(model, batchifier, scheduler_classes["basic"], train_data, dev_data, args.batch_size, freeze=True, gpu=args.gpu)

    logger.info(
        "Training StarCoder with %d/%d train/dev entities and %d/%d train/dev components", 
        len(train_ids), 
        len(dev_ids),
        train.num_components,
        dev.num_components
    )
    max_epochs = schema["meta"]["max_epochs"]
    batch_size = schema["meta"]["batch_size"] #1024

    #with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")) as p:
    with torch.autograd.profiler.profile(use_cuda=True, with_stack=True, profile_memory=True, record_shapes=True, enabled=args.profile) as p:
    #if True:
        #print(p.record_steps, p.step_rec_fn)
        #p._exit_actions = lambda : None
        for e in range(1, max_epochs + 1):

            train_loss, dev_loss = run_epoch(
                model,
                batchifier,
                optim,
                simple_loss_policy,
                train,
                dev,
                batch_size,
                args.gpu,
                #subselect=args.subselect,
                #train_property_dropout=args.train_property_dropout,
                #train_neuron_dropout=args.train_neuron_dropout,
                #dev_property_dropout=args.dev_property_dropout,
                #mask_properties=args.mask_properties
            )

            logger.info("Epoch %d: comparable train/dev loss = %.4f/%.4f",
                         e,
                         dev.num_entities * (train_loss / train.num_entities),
                         dev_loss,
            )
            reduce_rate, early_stop, new_best = sched.step(dev_loss)
            if new_best or best_dev_loss == None:
                logger.info("New best dev loss: %.3f", dev_loss)
                best_dev_loss = dev_loss
                #if best_state == None:
                best_state = {k : v.clone().detach().cpu() for k, v in model.state_dict().items()}

            if reduce_rate == True:
                model.load_state_dict(best_state)
            if early_stop == True:
                logger.info("Stopping early after no improvement for %d epochs", sched.early_stop)
                #if torch.isnan(best_dev_loss) or local_best_dev_loss < global_best_dev_loss:
                #    best_trace = current_trace
                #    gbd = "inf" if torch.isnan(global_best_dev_loss) else global_best_dev_loss / dev_data.num_entities
                #    logger.info("Random run #{}'s loss is current best ({:.4} < {:.4})".format(restart + 1, local_best_dev_loss, gbd))
                #    global_best_dev_loss = local_best_dev_loss
                #    global_best_state = local_best_state
                break
            #p.step()
            #elif e == max_epochs:
            #    logger.info("Stopping after reaching maximum epochs")
            #    break
                #if torch.isnan(best_dev_loss) or best_dev_loss < global_best_dev_loss:
                    #best_trace = current_trace
                #    gbd = "inf" if torch.isnan(global_best_dev_loss) else global_best_dev_loss / dev_data.num_entities
                #    logger.info("Random run #{}'s loss is current best ({:.4} < {:.4})".format(restart + 1, local_best_dev_loss / dev_data.num_entities, gbd))
                #    global_best_dev_loss = local_best_dev_loss
                #    global_best_state = local_best_state
    if args.profile:
        print(p.key_averages(group_by_stack_n=1).table(sort_by="cuda_time_total"))
        with open("trace.pkl.gz", "wb") as ofd:
            pickle.dump(p, ofd)
        #p.export_chrome_trace("trace.out")
    logger.info("Final dev loss of {:.4}".format(best_dev_loss))
    model.load_state_dict(best_state)    
    with gzip.open(args.model_output, "wb") as ofd:
        torch.save(model, ofd)

