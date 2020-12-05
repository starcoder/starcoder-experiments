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
import torch
from torch import Tensor
from starcoder.registry import property_model_classes, batchifier_classes, property_classes, scheduler_classes
from starcoder.ensemble_model import GraphAutoencoder
from starcoder.dataset import Dataset
from starcoder.schema import Schema
from starcoder.batchifier import Batchifier
from starcoder.property import CategoricalProperty
from starcoder.random import random
from starcoder.utils import run_epoch, simple_loss_policy
from torch.autograd import set_detect_anomaly
set_detect_anomaly(True)
from typing import Dict, List, Any, Tuple
import os

logger = logging.getLogger("train_model")

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input-related
    parser.add_argument("--data", dest="data", help="Input data file")
    parser.add_argument("--train", dest="train", help="Train split components file")
    parser.add_argument("--dev", dest="dev", help="Dev split components file")
    
    # model-related
    parser.add_argument("--depth", dest="depth", type=int, default=0, help="Graph-structure depth to consider")
    parser.add_argument("--embedding_size", type=int, dest="embedding_size", default=32, help="Size of embeddings")
    parser.add_argument("--hidden_size", type=int, dest="hidden_size", default=32, help="Size of embeddings")
    parser.add_argument("--rnn_max_decode", type=int, dest="rnn_max_decode", default=100, help="")
    parser.add_argument("--train_neuron_dropout", type=float, dest="train_neuron_dropout", default=0.0, help="Size of embeddings")
    parser.add_argument("--train_property_dropout", type=float, dest="train_property_dropout", default=0.0, help="Size of embeddings")
    parser.add_argument("--dev_property_dropout", type=float, dest="dev_property_dropout", default=0.0, help="Size of embeddings")
    parser.add_argument("--autoencoder_shapes", type=int, default=[], dest="autoencoder_shapes", nargs="*", help="Autoencoder layer sizes")
    parser.add_argument("--ae_loss", dest="ae_loss", default=False, action="store_true", help="Optimize autoencoder loss directly")
    parser.add_argument("--depthwise_boost", dest="depthwise_boost", default="none", choices=["none", "residual", "reconstruction"],
                        help="Method to address vanishing gradient in stacked autoencoders")
    
    # training-related
    parser.add_argument("--batchifier_class", dest="batchifier_class", default="sample_entities", choices=batchifier_classes.keys(),
                        help="Batchifier class")
    parser.add_argument("--max_epochs", dest="max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--patience", dest="patience", type=int, default=None, help="LR scheduler patience (default: no scheduler)")
    parser.add_argument("--early_stop", dest="early_stop", type=int, default=None, help="Early stop")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", dest="momentum", type=float, default=None, help="Momentum for SGD (default: Adam)")
    parser.add_argument("--gpu", dest="gpu", default=False, action="store_true", help="Use GPU")
    parser.add_argument("--subselect", dest="subselect", default=False, action="store_true", help="Subselect graph components to fit GPU")
    parser.add_argument("--random_restarts", dest="random_restarts", default=0, type=int, help="Number of random restarts to perform")

    # output-related
    parser.add_argument("--model_output", dest="model_output", help="Model output file")
    parser.add_argument("--trace_output", dest="trace_output", help="Trace output file")

    # miscellaneous
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int, help="Random seed")
    parser.add_argument("--log_level", dest="log_level", default="ERROR", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], help="Logging level")
    args, rest = parser.parse_known_args()
    batchifier = batchifier_classes[args.batchifier_class](rest)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )
    
    if isinstance(args.random_seed, int):
        logger.info("Setting random seed to %d across the board", args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed) # type: ignore
        torch.cuda.manual_seed(args.random_seed) # type: ignore
        torch.backends.cudnn.deterministic = True # type: ignore
        torch.backends.cudnn.benchmark = False # type: ignore
        numpy.random.seed(args.random_seed)
    with gzip.open(args.data, "rb") as ifd:
        data = pickle.load(ifd) # type: ignore
        
    logger.info("Loaded dataset: %s", data)
    logger.info("Dataset has schema: %s", data.schema)

    with gzip.open(args.train, "rt") as ifd:
        train_ids = [x[:-1] for x in ifd]

    with gzip.open(args.dev, "rt") as ifd:
        dev_ids = [x[:-1] for x in ifd]
        
    train_data = data.subselect_entities(train_ids)
    dev_data = data.subselect_entities(dev_ids)

    global_best_dev_loss = torch.tensor(numpy.nan)
    global_best_state = None
    best_trace: List[Dict[str, Any]]
    for restart in range(args.random_restarts + 1):

        local_best_dev_loss = torch.tensor(numpy.nan)
        local_best_state: Dict[str, Tensor]
        current_trace = []
        
        model = GraphAutoencoder(schema=data.schema,
                                 depth=args.depth,
                                 autoencoder_shapes=args.autoencoder_shapes,
                                 reverse_relationships=True,
                                 train_neuron_dropout=args.train_neuron_dropout,
                                 depthwise_boost=args.depthwise_boost,
        )        
        if args.gpu:
            model.cuda()
            logger.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                         torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)
            
        logger.info("Model: %s", model)
        logger.info("Model has %d parameters", model.parameter_count)

        model.init_weights()
        optim: Optimizer
        if args.momentum != None:
           logger.info("Using SGD with momentum")
           optim = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        else:
           logger.info("Using Adam")        
           optim = Adam(model.parameters(), lr=args.learning_rate)
        sched = scheduler_classes["basic"](args.early_stop, optim, patience=args.patience, verbose=True) if args.patience != None else None

        logger.info("Training StarCoder with %d/%d train/dev entities and %d/%d train/dev components and batch size %d", 
                     len(train_ids), 
                     len(dev_ids),
                     train_data.num_components,
                     dev_data.num_components,
                     args.batch_size)
        
        for e in range(1, args.max_epochs + 1):

            train_loss, train_loss_by_property, dev_loss, dev_loss_by_property, train_score_by_property, dev_score_by_property = run_epoch(
                model,
                batchifier,
                optim,
                simple_loss_policy,
                train_data,
                dev_data,
                args.batch_size, 
                args.gpu,
                subselect=args.subselect,
                train_property_dropout=args.train_property_dropout,
                train_neuron_dropout=args.train_neuron_dropout,
                dev_property_dropout=args.dev_property_dropout,
            )
            trace = {
                "iteration" : e,
                "losses" : {
                },
                "scores" : {}
            }

            current_trace.append(trace)
            
            logger.info("Random start %d, Epoch %d: comparable train/dev loss = %.4f/%.4f",
                         restart + 1,
                         e,
                         dev_data.num_entities * (train_loss / train_data.num_entities),
                         dev_loss,
            )
            reduce_rate, early_stop, new_best = sched.step(dev_loss)
            if new_best:
                logger.info("New best dev loss: %.3f", dev_loss)
                local_best_dev_loss = dev_loss
                local_best_state = {k : v.clone().detach().cpu() for k, v in model.state_dict().items()}

            if reduce_rate == True:
                model.load_state_dict(local_best_state)
            if early_stop == True:
                logger.info("Stopping early after no improvement for %d epochs", args.early_stop)
                if torch.isnan(global_best_dev_loss) or local_best_dev_loss < global_best_dev_loss:
                    best_trace = current_trace
                    gbd = "inf" if torch.isnan(global_best_dev_loss) else global_best_dev_loss / dev_data.num_entities
                    logger.info("Random run #{}'s loss is current best ({:.4} < {:.4})".format(restart + 1, local_best_dev_loss, gbd))
                    global_best_dev_loss = local_best_dev_loss
                    global_best_state = local_best_state
                break
            elif e == args.max_epochs:
                logger.info("Stopping after reaching maximum epochs")
                if torch.isnan(global_best_dev_loss) or local_best_dev_loss < global_best_dev_loss:
                    best_trace = current_trace
                    gbd = "inf" if torch.isnan(global_best_dev_loss) else global_best_dev_loss / dev_data.num_entities
                    logger.info("Random run #{}'s loss is current best ({:.4} < {:.4})".format(restart + 1, local_best_dev_loss / dev_data.num_entities, gbd))
                    global_best_dev_loss = local_best_dev_loss
                    global_best_state = local_best_state                    

    logger.info("Final dev loss of {:.4}".format(global_best_dev_loss))
    
    with gzip.open(args.model_output, "wb") as ofd:
        torch.save((global_best_state, args, data.schema), ofd) # type: ignore

    with gzip.open(args.trace_output, "wt") as ofd:
        final_trace: Dict[str, Any] = {}
        for epoch in best_trace:
            for split_name, properties in epoch.items():
                final_trace[split_name] = final_trace.get(split_name, {})
        ofd.write(json.dumps(best_trace)) # type: ignore

