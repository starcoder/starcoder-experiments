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
from starcoder.registry import field_model_classes, batchifier_classes, field_classes, scheduler_classes
from starcoder.ensemble_model import GraphAutoencoder
from starcoder.dataset import Dataset
from starcoder.schema import Schema
from starcoder.batchifier import Batchifier
from starcoder.field import CategoricalField
from starcoder.random import random
from starcoder.utils import run_epoch, simple_loss_policy
from torch.autograd import set_detect_anomaly
set_detect_anomaly(True)
from typing import Dict, List, Any, Tuple
import os

logger = logging.getLogger(__name__)

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
    parser.add_argument("--hidden_dropout", type=float, dest="hidden_dropout", default=0.0, help="Size of embeddings")
    parser.add_argument("--field_dropout", type=float, dest="field_dropout", default=0.0, help="Size of embeddings")
    parser.add_argument("--autoencoder_shapes", type=int, default=[], dest="autoencoder_shapes", nargs="*", help="Autoencoder layer sizes")
    parser.add_argument("--mask", dest="mask", default=[], nargs="*", help="Fields to mask")
    parser.add_argument("--ae_loss", dest="ae_loss", default=False, action="store_true", help="Optimize autoencoder loss directly")
    
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
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    
    args, rest = parser.parse_known_args()
    batchifier = batchifier_classes[args.batchifier_class](rest)
    mask_tests = [eval(l) for l in args.mask]

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if isinstance(args.random_seed, int):
        logger.info("Setting random seed to %d across the board", args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed) # type: ignore
        torch.cuda.manual_seed(args.random_seed) # type: ignore
        torch.backends.cudnn.deterministic = True # type: ignore
        torch.backends.cudnn.benchmark = False # type: ignore
        numpy.random.seed(args.random_seed)
        #starcoder.random.seed(args.random_seed)
        #os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    with gzip.open(args.data, "rb") as ifd:
        data = pickle.load(ifd) # type: ignore
        
    logger.info("Loaded dataset: %s", data)
    logger.info("Dataset has schema: %s", data.schema)

    with gzip.open(args.train, "rt") as ifd:
        train_ids = [x[:-1] for x in ifd]
        #pickle.load(ifd) # type: ignore

    with gzip.open(args.dev, "rt") as ifd:
        dev_ids = [x[:-1] for x in ifd]
        #dev_ids = pickle.load(ifd) # type: ignore
        
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

            train_loss, train_loss_by_field, dev_loss, dev_loss_by_field = run_epoch(model,
                                                                                     batchifier,
                                                                                     optim,
                                                                                     simple_loss_policy,
                                                                                     train_data,
                                                                                     dev_data,
                                                                                     args.batch_size, 
                                                                                     args.gpu,
                                                                                     mask_tests,
                                                                                     args.subselect
            )

            current_trace.append({"train" : train_loss_by_field, "dev" : dev_loss_by_field})
            
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
        #print(best_trace)
        final_trace: Dict[str, Any] = {} #"train" : {}, "dev" : {}}
        for epoch in best_trace:
            for split_name, fields in epoch.items():
                final_trace[split_name] = final_trace.get(split_name, {})
                #for field, loss in fields.items():
                #    final_trace[split_name][field.type_name] = final_trace[split_name].get(field.type_name, {})
                #    final_trace[split_name][field.type_name][field.name] = final_trace[split_name][field.type_name].get(field.name, [])
                #    final_trace[split_name][field.type_name][field.name].append(loss.item())
                    #final_trace["train"][field.type_name] = final_trace["train"].get(field.type_name, {})
                    #final_trace["train"][field.type_name][field.name] = final_trace["train"][field.type_name].get(field.name, [])
                    #train_loss = [x.tolist() for x in train_loss]
                    #final_trace["train"][field.type_name][field.name].append(train_loss)
        #for item in best_trace[1]:
        #    for field, dev_loss in item.items():
        #        final_trace["dev"][field.type_name] = final_trace["dev"].get(field.type_name, {})
        #        final_trace["dev"][field.type_name][field.name] = final_trace["dev"][field.type_name].get(field.name, [])
        #        dev_loss = sum(sum([x.tolist() for x in dev_loss], []))
        #        final_trace["dev"][field.type_name][field.name].append(dev_loss)
        ofd.write(json.dumps(final_trace)) # type: ignore

