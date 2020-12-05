import argparse
import re
import gzip
import json
import gensim
import logging
from gensim.models.ldamodel import LdaModel  # type: ignore[import]
from gensim.corpora.dictionary import Dictionary  # type: ignore[import]
from gensim.corpora.textcorpus import remove_stopwords  # type: ignore[import]
from gensim.utils import simple_preprocess  # type: ignore[import]

stopwords = set(['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'kg', 'km', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'make', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'quite', 'rather', 're', 'really', 'regarding', 'same', 'say', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'unless', 'until', 'up', 'upon', 'us', 'used', 'using', 'various', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'])

logger = logging.getLogger("train_topic_model")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data", help="Input file")
    parser.add_argument("--schema", dest="schema", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--num_topics", dest="num_topics", default=10, type=int)
    parser.add_argument("--alpha", dest="alpha", default="symmetric")
    parser.add_argument("--passes", dest="passes", default=1, type=int)
    parser.add_argument("--iterations", dest="iterations", default=50, type=int)
    parser.add_argument("--update_every", dest="update_every", default=1, type=int)
    parser.add_argument("--log_level", dest="log_level", default="ERROR", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )

    
    with open(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())

    docs = []
    vocab = Dictionary()
    logger.info("Reading in the data file...")
    with gzip.open(args.data, "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            tokens = []
            for k, v in j.items():
                property_type = schema["properties"].get(k, {}).get("type", None)
                if property_type == "text" and v != "":
                    tokens += ["word: {}".format(x) for x in v.lower().split() if x not in stopwords]
                elif property_type in ["categorical", "boolean"] and v != "":
                    tokens.append("{}: {}".format(k, v))

            
            docs.append(
                (
                    j["id"],
                    vocab.doc2bow(
                        remove_stopwords(
                            tokens
                        ),
                        allow_update=True,
                    ),
                )
            )
    logger.info("- done.")

    logger.info("Training an LDA model...")
    model = LdaModel(
        corpus=[d for _, d in docs],
        num_topics=args.num_topics,
        passes=args.passes,
        alpha=args.alpha,
        update_every=args.update_every,
        iterations=args.iterations,
        id2word=vocab,
    )
    logger.info("- done.")

    topic_reps = [
        ",".join([w for w, _ in model.show_topic(i)]) for i in range(args.num_topics)
    ]

    data = []
    topics = model.get_topics()
    with gzip.open(args.output, "wt") as ofd:
        for tid, scores in enumerate(topics):
            for wid, score in enumerate(scores):
                tp, word = re.match(r"^(.*): (.*)$", vocab[wid]).groups()
                ofd.write(
                    json.dumps(
                        {
                            "topic" : tid,
                            "type" : tp,
                            "word" : word,
                            "value" : float(score)
                        }
                    ) + "\n")
