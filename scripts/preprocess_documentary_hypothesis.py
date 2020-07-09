import argparse
import tarfile
import xml.etree.ElementTree as et
import gzip
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    source_lookup = {}
    entities = {"tanach" : {"entity_type" : "document", "document_name" : "tanach"}}
    with tarfile.open(args.inputs[0], "r:gz") as ifd:
        for mem in ifd.getmembers():
            if mem.isfile():
                content = ifd.extractfile(mem)
                xml = et.parse(content)
                for s in xml.findall(".//marks/mark"):
                    code = s.find(".//code").text
                    desc = s.find(".//description").text
                    source_lookup[code] = desc
                book_name = xml.find(".//tanach/book/names/name").text
                book_id = book_name
                entities[book_id] = {"entity_type" : "book", "book_name" : book_name}
                for chapter in xml.findall(".//tanach/book/c"):
                    chapter_num = int(chapter.attrib["n"])
                    chapter_id = "{}_{}".format(book_name, chapter_num)
                    entities[chapter_id] = {"entity_type" : "chapter", "chapter_number" : chapter_num, "from_book" : book_id}

                    for verse in chapter.findall(".//v"):
                        verse_num = int(verse.attrib["n"])
                        verse_id = "{}_{}_{}".format(book_name, chapter_num, verse_num)
                        words = [w.text for w in verse.findall(".//w")]
                        entities[verse_id] = {"entity_type" : "verse",
                                              "words" : " ".join(words),
                                              "verse_number" : verse_num,
                                              "from_chapter" : chapter_id}
                        if "s" in verse.attrib:
                            source_name = verse.attrib["s"]
                            source_id = "source_{}".format(source_name)
                            entities[source_id] = {"entity_type" : "source", "source_description" : source_lookup[source_name]}
                            entities[verse_id]["from_source"] = source_id

    with gzip.open(args.output, "wt") as ofd:
        for eid, entity in entities.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")
