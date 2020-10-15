import gzip
import logging
from geopy.geocoders import Photon as Service

logger = logging.getLogger(__name__)

class Geocoder(object):
    def __init__(self, cache_file, retries=5):
        self.retries = retries
        self.cache = {}
        if cache_file != None:
            with gzip.open(cache_file, "rt") as ifd:
                for line in ifd:
                    toks = line.strip().split("\t")
                    lat = float(toks[0])
                    lon = float(toks[1])
                    text = "\t".join(toks[2:])
                    self.cache[text] = {"latitude" : lat, "longitude" : lon}
        self._geocoder = Service(timeout=5)

    def __call__(self, text):
        if text not in self.cache:
            logger.info("Unknown location: {}".format(text))
            for i in range(self.retries):
                try:
                    x = self._geocoder.geocode(text)
                    if x != None:
                        self.cache[text] = {"latitude" : x.latitude, "longitude" : x.longitude}
                        break
                except:
                    x = None
            if x == None:
                self.cache[text] = None
        retval = self.cache.get(text, None)
        if not retval == None:
            retval["text"] = text
        return retval
