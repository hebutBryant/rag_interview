import os

FREEBASE_ORIGIN_DIR = "/home/hdd/depcache/dataset/FastRDFStore"
FREEBASE_DIR = "/home/hdd/depcache/dataset/FastRDFStore/process_data"


def get_id2name():

    input_file = os.path.join(FREEBASE_ORIGIN_DIR, "fb_en.txt")
    output_file = os.path.join(FREEBASE_DIR, "id2name.txt")

    if os.path.exists(output_file):
        print(f"{output_file} exists, skip!")
        return

    f_out = open(output_file, "w")

    relation = "type.object.name"
    with open(input_file, "r") as fp:
        for line in fp.readlines():
            split_line = line.strip().split("\t")
            if split_line[1] == relation:
                f_out.write(line)
    f_out.close()

    print(f"{output_file} is done!")


def manual_filter_rel():
    filter_domain = [
        "music.release",
        "authority.musicbrainz",
        "22-rdf-syntax-ns#type",
        "book.isbn",
        "common.licensed_object",
        "tv.tv_series_episode",
        "type.namespace",
        "type.content",
        "type.permission",
        "type.object.key",
        "type.object.permission",
        "type.type.instance",
        "topic_equivalent_webpage",
        "dataworld.freeq",
    ]
    filter_set = set(filter_domain)

    input_file = os.path.join(FREEBASE_ORIGIN_DIR, "fb_en.txt")
    output_file = os.path.join(FREEBASE_DIR, "manual_fb_filter.txt")

    if os.path.exists(output_file):
        print(f"{output_file} exists, skip!")
        return

    f_in = open(input_file)
    f_out = open(output_file, "w")
    num_line = 0
    num_reserve = 0
    for line in f_in:
        splitline = line.strip().split("\t")
        num_line += 1
        if len(splitline) < 3:
            continue
        rel = splitline[1]
        flag = False

        for domain in filter_set:
            if domain in rel:
                flag = True
                break

        if flag:
            continue
        f_out.write(line)
        num_reserve += 1
        if num_line % 10_000_000 == 0:
            print(f"{num_line=}, {num_reserve=}")

    f_in.close()
    f_out.close()
    print(f"{num_line=}, {num_reserve=}")
    print(f"{output_file} is done!")


def is_ent(tp_str):
    if len(tp_str) < 3:
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        print(tp_str)
        return True
    return False


def find_entity(sparql_str):
    str_lines = sparql_str.split("\n")
    ent_set = set()
    for line in str_lines[1:]:
        if "ns:" not in line:
            continue

        spline = line.strip().split(" ")
        for item in spline:
            ent_str = item[3:].replace("(", "")
            ent_str = ent_str.replace(")", "")
            if is_ent(ent_str):
                ent_set.add(ent_str)

    return ent_set


def abandon_rels(relation):
    if (
        relation == "type.object.type"
        or relation == "type.object.name"
        or relation.startswith("type.type.")
        or relation.startswith("common.")
        or relation.startswith("freebase.")
        or "sameAs" in relation
        or "sameas" in relation
    ):
        return True


def filter_rel():

    input_file = os.path.join(FREEBASE_DIR, "manual_fb_filter.txt")
    output_file = os.path.join(FREEBASE_DIR, "rel_filter.txt")

    if os.path.exists(output_file):
        print(f"{output_file} exists, skip!")
        return

    f_in = open(input_file)
    f_out = open(output_file, "w")
    num_line = 0
    num_reserve = 0

    for line in f_in:
        splitline = line.strip().split("\t")
        num_line += 1
        if len(splitline) < 3:
            continue
        rel = splitline[1]
        if abandon_rels(rel):
            continue
        f_out.write(line)
        num_reserve += 1
        if num_line % 10_000_000 == 0:
            print(f"{num_line=}, {num_reserve=}")

    f_in.close()
    f_out.close()
    print(f"{num_line=}, {num_reserve=}")


def get_triplets():
    input_file = os.path.join(FREEBASE_DIR, "rel_filter.txt")

    edges = []
    nodes = set()
    cnt = 0
    rels = set()

    with open(input_file, "r") as fp:
        for line in fp:
            split = line.strip().split("\t")
            assert len(split) == 3
            head, rel, tail = line.strip().split("\t")
            nodes.add(head)
            nodes.add(tail)
            edges.append((head, tail, rel))
            rels.add(rel)

            cnt += 1
            if cnt % 10_000_000 == 0:
                print(f"read {cnt} edges.")

    print("read from {input_file}, done!")
    print(f"{len(nodes)} nodes and {len(edges)} edges, {len(rels)} relations.")

    rels = list(rels)
    nodes = list(nodes)

    return nodes, edges, rels


def process_data():

    os.makedirs(FREEBASE_DIR, exist_ok=True)

    get_id2name()

    manual_filter_rel()

    filter_rel()


if __name__ == "__main__":

    process_data()

    get_triplets()
