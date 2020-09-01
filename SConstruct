import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
from hashlib import md5
#import starcoder
import steamroller


# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle


vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 100),
    ("RANDOM_COMPONENTS", "", 1000),
    ("MINIMUM_CONSTANTS", "", 1),
    ("MAXIMUM_CONSTANTS", "", 4),
    ("TRAIN_PROPORTION", "", 0.9),
    ("DEV_PROPORTION", "", 0.1),
    ("MAX_CATEGORICAL", "", 50),
    ("MAX_SEQUENCE_LENGTH", "", 100),
    ("RANDOM_LINE_COUNT", "", 1000),
    ("MAX_COLLAPSE", "", 0),
    ("MAX_SEQUENCE_LENGTH", "", 32),
    ("LOG_LEVEL", "", "INFO"),
    ("LINE_COUNT", "", 1000),
    ("BATCH_SIZE", "", 128),
    ("MAX_EPOCHS", "", 10),
    ("LEARNING_RATE", "", 0.001),
    ("RANDOM_RESTARTS", "", 0),
    ("ELASTIC_HOST", "", "localhost"),
    ("ELASTIC_PORT", "", 5601),
    ("MOMENTUM", "", None),
    ("EARLY_STOP", "", 10),
    ("PATIENCE", "", 5),
    ("HIDDEN_SIZE", "", 32),
    ("EMBEDDING_SIZE", "", 32),
    ("AUTOENCODER_SHAPES", "", [32, 16]),
    ("CLUSTER_REDUCTION", "", 0.5),
    BoolVariable("USE_GPU", "", False),
    BoolVariable("USE_GRID", "", False),
    ("GPU_PREAMBLE", "", "module load cuda10.1/toolkit"),
    ("SPLIT_PROPORTIONS", "", [("train", 0.80), ("dev", 0.10), ("test", 0.10)]),
    ("DEPTH", "", 1),
    ("SPLITTER_CLASS", "", "sample_components"),
    ("BATCHIFIER_CLASS", "", "sample_components"),
    ("SLAVE_TRADE_PATH", "", "${DATA_PATH}/slavery"),
    ("MAROON_PATH", "", "${DATA_PATH}/maroon_ads"),
    ("FLUENCY_PATH", "", "${DATA_PATH}/russian_fluency"),
    ("SENTIMENT_PATH", "", "${DATA_PATH}/stanford_sentiment_treebank"),
    ("SMOKING_AND_VAPING_PATH", "", "${DATA_PATH}/smoking_and_vaping.txt.bz2"),
    ("ENTERTAINING_PATH", "", "${DATA_PATH}/entertaining_america"),
    ("DH_PATH", "", "${DATA_PATH}/documentary_hypothesis"),
    ("ME_PATH", "", "${DATA_PATH}/middle_english"),    
    ("AFFICHES_PATH", "", "${DATA_PATH}/affiches_americaines"),
    ("ROYAL_INSCRIPTIONS_PATH", "", "${DATA_PATH}"),
    ("WALS_PATH", "", "${DATA_PATH}"),
    ("TWITTER_LID_PATH", "", "${DATA_PATH}/twitter_lid"),
    ("WOMEN_WRITERS_PATH", "", "${DATA_PATH}"),
    ("PARIS_TAX_ROLLS_PATH", "", "${DATA_PATH}"),
    ("LITBANK_PATH", "", "${DATA_PATH}/litbank"),
    ("MULTIMODAL_WIKIPEDIA_PATH", "", "${DATA_PATH}/multimodal_wikipedia"),
    ("DATA_PATH", "", ""),
    ("EXPERIMENTS", "", {}),
    ("ELASTIC_HOST", "", "localhost"),
    ("ELASTIC_PORT", "", 9200),
    ("ELASTIC_USER", "", "elastic"),
    ("ELASTIC_PASSWORD", "", ""),
    ("KIBANA_HOST", "", "localhost"),
    ("KIBANA_PORT", "", 5601),
    ("KIBANA_USER", "", "elastic"),
    ("KIBANA_PASSWORD", "", ""),
    ("DJANGO_USER", "", "admin"),
    ("DJANGO_PASSWORD", "", "admin"),
    ("DJANGO_EMAIL", "", "nothing@nothing.com"),
    BoolVariable("ELASTIC_UPLOAD", "", False),
    ("SERVER_HOST", "", "localhost"),
    ("SERVER_PORT", "", 8080),
)


env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", steamroller.generate])


env.Append(BUILDERS={"PreprocessArithmetic" : env.Builder(**env.ActionMaker("python",
                                                                            "scripts/preprocess_arithmetic.py",
                                                                            "--output ${TARGETS[0]} --components ${RANDOM_COMPONENTS} --minimum_constants ${MINIMUM_CONSTANTS} --maximum_constants ${MAXIMUM_CONSTANTS}")),
                     "PreprocessTargetedSentiment" : env.Builder(**env.ActionMaker("python",
                                                                                   "scripts/preprocess_targeted_sentiment.py",
                                                                                   "${SOURCES} --output ${TARGETS[0]}")),
                     "PreprocessMaroonAds" : env.Builder(**env.ActionMaker("python",
                                                                           "scripts/preprocess_maroon_ads.py",
                                                                           "${SOURCES[0]} --output ${TARGETS[0]}")),
                     "PreprocessFluency" : env.Builder(**env.ActionMaker("python",
                                                                         "scripts/preprocess_fluency.py",
                                                                         "${SOURCES} --output ${TARGETS[0]}")),
                     "PreprocessSmokingAndVaping" : env.Builder(**env.ActionMaker("python",
                                                                                  "scripts/preprocess_reddit.py",
                                                                                  "${SOURCES} --output ${TARGETS[0]}")),                     
                     "PreprocessMiddleEnglish" : env.Builder(**env.ActionMaker("python",
                                                                               "scripts/preprocess_middle_english.py",
                                                                               "${SOURCES} --output ${TARGETS[0]}")),
                     "PreprocessDocumentaryHypothesis" : env.Builder(**env.ActionMaker("python",
                                                                                       "scripts/preprocess_documentary_hypothesis.py",
                                                                                       "${SOURCES} --output ${TARGETS[0]}")),
                     "PreprocessAffichesAmericaines" : env.Builder(**env.ActionMaker("python",
                                                                                     "scripts/preprocess_affiches_americaines.py",
                                                                                     "${SOURCES} --output ${TARGETS[0]}")),
                     "PreprocessParisTaxRolls" : env.Builder(**env.ActionMaker("python",
                                                                               "scripts/preprocess_paris_tax_rolls.py",
                                                                               "${SOURCES} --output ${TARGETS[0]}")),
                     "PreprocessEntertainingAmerica" : env.Builder(**env.ActionMaker("python",
                                                                                     "scripts/preprocess_entertaining_america.py",
                                                                                     "${SOURCES} --output ${TARGETS[0]}")),
                     "PreprocessWomenWriters" : env.Builder(**env.ActionMaker("python",
                                                                              "scripts/preprocess_women_writers.py",
                                                                              "${SOURCES} --output ${TARGETS[0]}")),
                     "PreprocessRoyalInscriptions" : env.Builder(**env.ActionMaker("python",
                                                                                   "scripts/preprocess_royal_inscriptions.py",
                                                                                   "${SOURCES} --output ${TARGETS[0]}")),
                     "PreprocessLinguisticLid" : env.Builder(**env.ActionMaker("python",
                                                                               "scripts/preprocess_linguistic_lid.py",
                                                                               "--output ${TARGETS[0]} ${SOURCES}")),
                     "PreprocessLitbank" : env.Builder(**env.ActionMaker("python",
                                                                         "scripts/preprocess_litbank.py",
                                                                         "--output ${TARGETS[0]} --input ${SOURCES[0]}")),
                     "PreprocessMultimodalWikipedia" : env.Builder(**env.ActionMaker("python",
                                                                                     "scripts/preprocess_multimodal_wikipedia.py",
                                                                                     "--output ${TARGETS[0]} --input ${SOURCES[0]}")),
                     "PreprocessPostAtlanticSlaveTrade" : env.Builder(**env.ActionMaker("python",
                                                                                        "scripts/preprocess_post_atlantic_slave_trade.py",
                                                                                        "--output ${TARGETS[0]} ${SOURCES}")),
                     "PrepareDataset" : env.Builder(**env.ActionMaker("python",
                                                               "scripts/prepare_dataset.py",
                                                               "--schema_output ${TARGETS[0]} --data_output ${TARGETS[1]} --data_input ${SOURCES[0]} --schema_input ${SOURCES[1]}",
                                                               other_deps=[])),
                     "SplitData" : env.Builder(**env.ActionMaker("python",
                                                                 "scripts/split_data.py",
                                                                 "--input ${SOURCES[0]} --proportions ${PROPORTIONS} --outputs ${TARGETS} --random_seed ${RANDOM_SEED} --splitter_class ${SPLITTER_CLASS} --shared_entity_types ${SHARED_ENTITY_TYPES}",
                                                                 other_deps=[],
                     )),
                     "TrainModel" : env.Builder(**env.ActionMaker("python",
                                                           "scripts/train_model.py",
                                                           "--data ${SOURCES[0]} --train ${SOURCES[1]} --dev ${SOURCES[2]} --model_output ${TARGETS[0]} --trace_output ${TARGETS[1]} ${'--gpu' if USE_GPU else ''} ${'--autoencoder_shapes ' + ' '.join(map(str, AUTOENCODER_SHAPES)) if AUTOENCODER_SHAPES != None else ''} ${'--mask ' + ' '.join(MASK) if MASK else ''} --log_level ${LOG_LEVEL} ${'--autoencoder' if AUTOENCODER else ''} --random_restarts ${RANDOM_RESTARTS} ${' --subselect ' if SUBSELECT==True else ''} --batchifier_class ${BATCHIFIER_CLASS} --shared_entity_types ${SHARED_ENTITY_TYPES}",
                                                           other_args=["DEPTH", "MAX_EPOCHS", "LEARNING_RATE", "RANDOM_SEED", "PATIENCE", "MOMENTUM", "BATCH_SIZE",
                                                                       "EMBEDDING_SIZE", "HIDDEN_SIZE", "FIELD_DROPOUT", "HIDDEN_DROPOUT", "EARLY_STOP"],
                                                                  USE_GPU=env["USE_GPU"],
                                                           )),
                     "ApplyModel" : env.Builder(**env.ActionMaker("python",
                                                                  "scripts/apply_model.py",
                                                                  "--model ${SOURCES[0]} --dataset ${SOURCES[1]} ${'--split ' + SOURCES[2].rstr() if len(SOURCES) == 3 else ''} --output ${TARGETS[0]} ${'--gpu' if USE_GPU else ''}",
                     )),
                     "UploadResults" : env.Builder(**env.ActionMaker("python",
                                                                  "scripts/upload_results.py",
                                                                  "--results ${SOURCES[0]} --log ${TARGETS[0]} --elastic_host ${ELASTIC_HOST} --elastic_port ${ELASTIC_PORT} --create_indices --overwrite_existing --experiment_id ${APPLY_CONFIG_ID} --experiment_name ${EXPERIMENT_NAME} --elastic_user ${ELASTIC_USER} --elastic_password ${ELASTIC_PASSWORD} --kibana_host ${KIBANA_HOST} --kibana_port ${KIBANA_PORT} --kibana_user ${KIBANA_USER} --kibana_password ${KIBANA_PASSWORD} --model ${SOURCES[1]}",
                     )),
                     #"ClusterEntities" : env.Builder(**env.ActionMaker("python",
                     #                                                  "scripts/cluster_entities.py",
                     #                                                  "--input ${SOURCES[0]} --schema ${SOURCES[1]} --output ${TARGETS[0]} --reduction ${CLUSTER_REDUCTION}")),
                     #"InspectClusters" : env.Builder(**env.ActionMaker("python",
                     #                                                  "scripts/inspect_clusters.py",
                     #                                                  "--input ${SOURCES[0]} --schema ${SOURCES[1]} --output ${TARGETS[0]}")),
                     "PlotTrace" : env.Builder(**env.ActionMaker("python",
                                                                 "scripts/plot_trace.py",
                                                                 "--input ${SOURCES[0]} --output ${TARGETS[0]}")),
                     "MakeServerConfig" : env.Builder(**env.ActionMaker("python",
                                                                        "scripts/make_server_config.py",
                                                                        "--models ${SOURCES} --names ${NAMES} --elastic_host ${ELASTIC_HOST} --elastic_port ${ELASTIC_PORT} --elastic_user ${ELASTIC_USER} --elastic_password ${ELASTIC_PASSWORD} --host ${SERVER_HOST} --port ${SERVER_PORT} --django_user ${DJANGO_USER} --django_password ${DJANGO_PASSWORD} --django_email ${DJANGO_EMAIL} --output ${TARGETS[0]}")),
                 },
           tools=["default"],
)


# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)


# and the command-printing function
env['PRINT_CMD_LINE_FUNC'] = print_cmd_line


# and how we decide if a dependency is out of date
env.Decider("timestamp-newer")


def run_experiment(env, experiment_config, **args):
    data = sum([env.Glob(env.subst(p)) for p in experiment_config.get("DATA_FILES", [])], [])
    schema = experiment_config.get("SCHEMA", None)
    title = experiment_name.replace("_", " ").title().replace(" ", "")
    data = getattr(env, "Preprocess{}".format(title))("work/${EXPERIMENT_NAME}/data.json.gz",
                                                      data, **args)

    # prepare the final spec and dataset
    observed_schema, dataset = env.PrepareDataset(["work/${EXPERIMENT_NAME}/schema.json.gz", "work/${EXPERIMENT_NAME}/dataset.pkl.gz"],
                                                  [data] + ([] if schema == None else [schema]),
                                                  **args)
    
    split_names = [n for n, _ in experiment_config.get("SPLIT_PROPORTIONS", env["SPLIT_PROPORTIONS"])]
    split_props = [p for _, p in experiment_config.get("SPLIT_PROPORTIONS", env["SPLIT_PROPORTIONS"])]    

    
    train, dev, test = env.SplitData(["work/${{EXPERIMENT_NAME}}/{0}.pkl.gz".format(n) for n in split_names], 
                                     dataset,
                                     **experiment_config,
                                     **args, RANDOM_SEED=0, PROPORTIONS=split_props)

    env.Alias("splits", [train, dev, test])
    
    # expand training configurations
    train_configs = [[]]
    for arg_name, values in experiment_config.get("TRAIN_CONFIG", {}).items():        
       train_configs = sum([[config + [(arg_name.upper(), v)] for config in train_configs] for v in values], [])
    train_configs = [dict(config) for config in train_configs]
    
    # expand apply configurations
    apply_configs = [[]]
    for arg_name, values in experiment_config.get("APPLY_CONFIG", {}).items():
       apply_configs = sum([[config + [(arg_name.upper(), v)] for config in apply_configs] for v in values], [])
    apply_configs = [dict(config) for config in apply_configs]
    
    results = []
    for config in train_configs:
        args["TRAIN_CONFIG_ID"] = md5(str(sorted(list(config.items()))).encode()).hexdigest()
        model, trace = env.TrainModel(["work/${EXPERIMENT_NAME}/model_${TRAIN_CONFIG_ID}.pkl.gz", 
                                       "work/${EXPERIMENT_NAME}/trace_${TRAIN_CONFIG_ID}.json.gz"],
                                      [dataset, train, dev],
                                      **args,
                                      **experiment_config,
                                      **config)
        env.PlotTrace("work/${EXPERIMENT_NAME}/traceplot_${TRAIN_CONFIG_ID}.png",
                      trace,
                      **args
        )
        for apply_config in apply_configs:
            config.update(apply_config)
            args["APPLY_CONFIG_ID"] = md5(str(sorted(list(config.items()))).encode()).hexdigest()
            output = env.ApplyModel("work/${EXPERIMENT_NAME}/${FOLD}/output_${APPLY_CONFIG_ID}.json.gz",
                                    [model, dataset],
                                    **args,
                                    **experiment_config,
                                    **config)
            if env["ELASTIC_UPLOAD"] == True:
                upload = env.UploadResults("work/${EXPERIMENT_NAME}/${FOLD}/upload_${APPLY_CONFIG_ID}.txt",
                                           [output, model],
                                           **args,
                                           **experiment_config,
                                           **config)
            #continue            
            #clusters = env.ClusterEntities("work/${EXPERIMENT_NAME}/${FOLD}/clusters_${APPLY_CONFIG_ID}.json.gz",
            #                               [output, schema],
            #                               **args,
            #                               **config)
            #inspect_clusters = env.InspectClusters("work/${EXPERIMENT_NAME}/${FOLD}/inspect_clusters_${APPLY_CONFIG_ID}.txt",
            #                                       [clusters, observed_schema],
            #                                       **args,
            #                                       **config)
    return model


env.AddMethod(run_experiment, "RunExperiment")


#
# Run all experiments
#
models = []
names = []
for experiment_name, experiment_config in env["EXPERIMENTS"].items():
    names.append(experiment_name)
    models.append(env.RunExperiment(experiment_config, EXPERIMENT_NAME=experiment_name))
env.MakeServerConfig("work/server_config.json", models, NAMES=names)
