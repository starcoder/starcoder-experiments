from traceback import format_exc
import json
import gzip
from io import BytesIO
import argparse
import requests
import bs4
import os.path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", 
        dest="url", 
        help="Input URL", 
        default="https://www.tdcj.texas.gov/death_row"
    )
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    entries = {}
    out_fname = os.path.join(args.output, "entries.json.gz")
    if os.path.exists(out_fname):
        with gzip.open(out_fname, "rt") as ifd:
            for line in ifd:
                entry = json.loads(line)
                entries[entry["execution_id"]] = entry

    p = bs4.BeautifulSoup(
        requests.get("{}/dr_executed_offenders.html".format(args.url), timeout=10).content, 
        "html.parser"
    )
    t = p.find("table", title="Table showing list of executed offenders")

    for r in t.find_all("tr")[1:]:
        entry = {}
        eid, info, statement, last, first, oid, age, date, race, county = r.find_all("td")
        entry["execution_id"] = eid.text.strip()
        entry["last_name"] = last.text.strip()
        entry["first_name"] = first.text.strip()
        entry["TDCJ"] = oid.text.strip()
        entry["age"] = age.text.strip()
        entry["date"] = date.text.strip()
        entry["race"] = race.text.strip()
        entry["county"] = county.text.strip()
        print(entry["first_name"], entry["last_name"])
        if entry["execution_id"] in entries:
            continue
        print(entry)
        try:
            info_url = "{}/{}".format(args.url, info.find("a").get("href").replace("death_row", ""))
            statement_url = "{}/{}".format(args.url, statement.find("a").get("href").replace("death_row", ""))            
            if info_url.endswith("html"):
                if "no_info_available" not in info_url:
                    ip = bs4.BeautifulSoup(requests.get(info_url, timeout=10).content, "html.parser")
                    table = ip.find("table", class_="table_deathrow")
                    try:
                        img_fname = table.find("img").get("src")
                        image_url = "{}/dr_info/{}".format(args.url, img_fname)
                        image = requests.get(image_url, timeout=10).content
                        with open(os.path.join(args.output, "images", img_fname), "wb") as ofd:
                            ofd.write(image)
                        entry["suspect_image"] = img_fname
                    except AttributeError as e:
                        pass
                    for tr in table.find_all("tr"):
                        try:
                            k, v = list(tr.find_all("td"))
                        except:
                            _, k, v = list(tr.find_all("td"))
                        entry[k.text] = v.text
                    for sp in ip.select("p > span"):
                        n = sp.text.strip()
                        v = sp.parent.text
                        v = v.replace(n, "").strip()
                        entry[n] = v
            else:
                image = requests.get(info_url, timeout=10).content
                img_fname = os.path.basename(info_url)
                with open(os.path.join(args.output, "images", img_fname), "wb") as ofd:
                    ofd.write(image)
                entry["docket_image"] = img_fname

            sp = bs4.BeautifulSoup(requests.get(statement_url, timeout=10).content, "html.parser").find_all("p")
            for i in range(len(sp) - 1):
               if "Last Statement" in sp[i].get_text():
                   entry["statement"] = sp[i + 1].get_text()
               elif "Date of Execution" in sp[i].get_text():
                   entry["date_of_execution"] = sp[i + 1].get_text()
        except Exception as e:
            print(e)
            print(format_exc())
            break
        entries[entry["execution_id"]] = entry

    with gzip.open(os.path.join(args.output, "entries.json.gz"), "wt") as ofd:
        for entry in entries.values():
            ofd.write(json.dumps(entry) + "\n")
