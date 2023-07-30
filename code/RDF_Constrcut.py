
from datetime import datetime
import hashlib
import os.path
import json
import re
from urllib.parse import quote
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, Literal, BNode, Namespace, RDF, URIRef, XSD, OWL, SKOS, DC, RDFS

ASKG = Graph()
ASKG_namespace_onto = Namespace("https://www.anu.edu.au/onto/scholarly#")
ASKG_namespace_data = Namespace("https://www.anu.edu.au/onto/scholarly/")
DOMO_namespace = Namespace("https://www.anu.edu.au/onto/domo#")
WIKIDATA_namespace = Namespace("http://www.wikidata.org/entity/")
TNNT_namespace = Namespace("https://soco.cecc.anu.edu.au/tool/TNNT#")

ASKG.bind("askg-onto", ASKG_namespace_onto)
ASKG.bind("askg-data", ASKG_namespace_data)
ASKG.bind("domo", DOMO_namespace)
ASKG.bind("wd", WIKIDATA_namespace)
ASKG.bind("skos", SKOS)
ASKG.bind("dc", DC)
ASKG.bind("tnnt", TNNT_namespace)

mode = "ssh"

tnnt_list = []
if mode == "ssh":
    json_file = "/home/users/u000000/askg/a-scholarly-kg/src/Papers/RDF/ASKG_wiki_dict.json"
else:
    json_file = "ASKG_wiki_dict.json"
with open(json_file, 'r', encoding='utf-8') as file:
    wiki_dict = json.load(file)

# blake2s hash function
def blake2s_hash(input_string, digest_size=7):
    input_bytes = input_string.encode('utf-8')
    hasher = hashlib.blake2s(digest_size=digest_size)
    hasher.update(input_bytes)
    hash_result = hasher.digest()
    return hash_result.hex()

# load the paper section location
def find_located_sec(pos_idx_from, pos_idx_to, paper_id):
    if mode == "ssh":
        loc_path = "/home/users/u7274475/askg/anu-scholarly-kg/src/Papers/ASKG_Paper_Dataset/PaperSectionLocation/paper_section_loc.json"
    else:
        loc_path = "../ASKG_Paper_Dataset/PaperSectionLocation/paper_section_loc.json"
    with open(loc_path) as f:
        loc_dict = json.load(f)

    section_loc = loc_dict.get(paper_id, 0)
    if section_loc != 0:
        for key in section_loc.keys():
            if pos_idx_from < section_loc[key][1] and pos_idx_from > section_loc[key][0]:
                return key
    else:
        return None

# construct KG for the paper
def construct_KG_Excerpt_Academic_Entity(ASKG, excerpt_result, paper_id, section_dict, paper_info):

    for key in excerpt_result.keys():
        for entity_key in excerpt_result[key]:
            wikidata_ID = excerpt_result[key][entity_key].get("wikidata_ID", "")
            wikidata_name = excerpt_result[key][entity_key].get("matched_words", "")
            position = excerpt_result[key][entity_key]["position"]
            occur_times = len(position)

            # construct the academic entity
            if wikidata_ID == "":
                academic_entity_str = quote(entity_key.lower().replace(" ", "_"), safe='')
            else:
                academic_entity_str = quote(entity_key.lower().replace(" ", "_"), safe='') + "-" +wikidata_ID
            academic_entity = ASKG_namespace_data[f"AcademicEntity-{academic_entity_str}"]

            # construct the Wikidata entity
            Wikidata_entity = WIKIDATA_namespace[f"{wikidata_ID}"]
            if (Wikidata_entity, None, None) not in ASKG:
                ltr_wikidata_name = Literal(wikidata_name, lang="en")
                ASKG.add((Wikidata_entity, RDFS.label, ltr_wikidata_name))
            if (academic_entity, None, None) not in ASKG:
                ltr_academic_entity_name = Literal(entity_key.lower(), datatype=XSD.string)
                ASKG.add((academic_entity, RDFS.label, ltr_academic_entity_name))
                if (academic_entity, OWL.sameAs, Wikidata_entity) not in ASKG and wikidata_ID != "":
                    ASKG.add((academic_entity, OWL.sameAs, Wikidata_entity))
            if (academic_entity, SKOS.broader, scientific_type_dict[key.upper()]) not in ASKG:
                ASKG.add((academic_entity, SKOS.broader, scientific_type_dict[key.upper()]))
            if (scientific_type_dict[key.upper()], SKOS.narrower, academic_entity) not in ASKG:
                ASKG.add((scientific_type_dict[key.upper()], SKOS.narrower, academic_entity))

            # construct the excerpt
            for i in range(occur_times):
                pos_idx_from = excerpt_result[key][entity_key]["position"][i][0]
                pos_idx_to = excerpt_result[key][entity_key]["position"][i][-1]
                sentence = excerpt_result[key][entity_key]["sentence"][i]

                # find the section location
                sec_loc = find_located_sec(pos_idx_from, pos_idx_to, paper_id)
                sec = section_dict[sec_loc]


                paper_title = paper_info["title"]
                excerpt_label_str = "Paper-" + f"{[paper_title]}" + " | " + "Section-" + f"{[sec_loc]}" + " | " + "Excerpt-" + f"{[pos_idx_from]}" + "-" + f"{[pos_idx_to]}"
                excerpt_label = Literal(excerpt_label_str, lang="en")
                hashed_excerpt = blake2s_hash(excerpt_label_str)
                excerpt_entity = ASKG_namespace_data[f"Excerpt-{hashed_excerpt}"]


                # ASKG.add((excerpt_entity, RDF.type, excerpt))
                ASKG.add((sec, rel_contains, excerpt_entity))
                ASKG.add((excerpt_entity, rel_mentions, academic_entity))
                ASKG.add((excerpt_entity, RDFS.label, excerpt_label))

                ltr_pos_idx_from = Literal(pos_idx_from, datatype=XSD.int)
                ltr_pos_idx_to = Literal(pos_idx_to, datatype=XSD.int)
                ltr_sentence = Literal(sentence, datatype=XSD.string)
                ASKG.add((excerpt_entity, rel_word_idx_from, ltr_pos_idx_from))
                ASKG.add((excerpt_entity, rel_word_idx_to, ltr_pos_idx_to))
                ASKG.add((excerpt_entity, rel_in_sent, ltr_sentence))
    return ASKG

# construct KG for the paper
def construct_KG_Paper_Abs(ASKG):

    global rel_hasSection
    rel_hasSection = ASKG_namespace_onto.hasSection

    section = ASKG_namespace_onto.Section
    domo_document_model_component = DOMO_namespace.DocumentModelComponent
    # ASKG.add((section, RDFS.subClassOf, domo_document_model_component))

    global paper
    paper = ASKG_namespace_onto.Paper
    domo_document = DOMO_namespace.Document

    global abstract, introduction, related_work, experiment, methodology, discussion
    abstract = ASKG_namespace_onto.Abstract
    introduction = ASKG_namespace_onto.Introduction
    related_work = ASKG_namespace_onto.RelatedWork
    experiment = ASKG_namespace_onto.Experiment
    methodology = ASKG_namespace_onto.Methodology
    discussion = ASKG_namespace_onto.Discussion

    # ASKG.add((abstract, RDFS.subClassOf, section))
    # ASKG.add((introduction, RDFS.subClassOf, section))
    # ASKG.add((related_work, RDFS.subClassOf, section))
    # ASKG.add((experiment, RDFS.subClassOf, section))
    # ASKG.add((methodology, RDFS.subClassOf, section))
    # ASKG.add((discussion, RDFS.subClassOf, section))

    global rel_domo_keyword
    rel_domo_keyword = DOMO_namespace.keyword
    global KeywordOfSection
    KeywordOfSection = ASKG_namespace_onto.KeywordOfSection
    # ASKG.add((section, rel_domo_keyword, KeywordOfSection))
    global rel_correspondsTo
    rel_correspondsTo = ASKG_namespace_onto.correspondsTo
    global Keyword
    Keyword = ASKG_namespace_onto.Keyword
    # ASKG.add((KeywordOfSection, rel_correspondsTo, Keyword))

    global excerpt
    excerpt = ASKG_namespace_onto.Excerpt
    global rel_contains
    rel_contains = ASKG_namespace_onto.contains
    # ASKG.add((section, rel_contains, excerpt))

    domo_content = DOMO_namespace.Content
    # ASKG.add((excerpt, RDFS.subClassOf, domo_content))

    global rel_mentions
    rel_mentions = ASKG_namespace_onto.mentions

    academic_entity = ASKG_namespace_onto.AcademicEntity
    # ASKG.add((excerpt, rel_mentions, academic_entity))

    scientific_type = ASKG_namespace_onto.ScientificType
    # ASKG.add((academic_entity, SKOS.broader, scientific_type))
    # ASKG.add((scientific_type, SKOS.narrower, academic_entity))
    # ASKG.add((academic_entity, RDFS.subClassOf, SKOS.Concept))
    # ASKG.add((scientific_type, RDFS.subClassOf, SKOS.Concept))

    global scientific_type_dict
    scientific_type_dict = {}
    scientific_type_dict["SOLUTION"] = ASKG_namespace_onto.Solution
    scientific_type_dict["RESEARCH_PROBLEM"] = ASKG_namespace_onto.ResearchProblem
    scientific_type_dict["METHOD"] = ASKG_namespace_onto.Method
    scientific_type_dict["DATASET"] = ASKG_namespace_onto.Dataset
    scientific_type_dict["TOOL"] = ASKG_namespace_onto.Tool
    scientific_type_dict["LANGUAGE"] = ASKG_namespace_onto.Language
    scientific_type_dict["RESOURCE"] = ASKG_namespace_onto.Resource

    for key in scientific_type_dict.keys():
        ASKG.add((scientific_type_dict[key], RDF.type, scientific_type))

    global rel_word_idx_from, rel_word_idx_to, rel_in_sent
    rel_word_idx_from = ASKG_namespace_onto.wordIndexFrom
    rel_word_idx_to = ASKG_namespace_onto.wordIndexTo
    rel_in_sent = ASKG_namespace_onto.inSentence

    global rel_CharIndexFrom, rel_CharIndexTo
    rel_CharIndexFrom = ASKG_namespace_onto.charIndexFrom
    rel_CharIndexTo = ASKG_namespace_onto.charIndexTo

    global tnnt_dict
    tnnt_dict = {}
    tnnt_dict["PERSON"] = TNNT_namespace.Person
    tnnt_dict["NORP"] = TNNT_namespace.NORP
    tnnt_dict["FAC"] = TNNT_namespace.FAC
    tnnt_dict["ORG"] = TNNT_namespace.Organization
    tnnt_dict["GPE"] = TNNT_namespace.GPE
    tnnt_dict["LOC"] = TNNT_namespace.Location
    tnnt_dict["PRODUCT"] = TNNT_namespace.Product
    tnnt_dict["EVENT"] = TNNT_namespace.Event
    tnnt_dict["WORK_OF_ART"] = TNNT_namespace.WorkOfArT
    tnnt_dict["LAW"] = TNNT_namespace.Law
    tnnt_dict["LANGUAGE"] = TNNT_namespace.Language
    tnnt_dict["DATE"] = TNNT_namespace.Date
    tnnt_dict["TIME"] = TNNT_namespace.Time
    tnnt_dict["PERCENT"] = TNNT_namespace.Percent
    tnnt_dict["MONEY"] = TNNT_namespace.Money
    tnnt_dict["QUANTITY"] = TNNT_namespace.Quantity
    tnnt_dict["ORDINAL"] = TNNT_namespace.Ordinal
    tnnt_dict["CARDINAL"] = TNNT_namespace.Cardinal

    return ASKG

def get_paper_info(json_paper_info, paper_id):
    info_paper_id = json_paper_info.get(paper_id, 0)

    paper_info = {"title": "", "doi": "", "link": "", "authors": ""}
    if info_paper_id != 0:
        paper_info["title"] = info_paper_id.get("title")
        paper_info["doi"] = info_paper_id.get("doi")
        paper_info["link"] = info_paper_id.get("link")
        paper_info["authors"] = info_paper_id.get("author", ["None"])

        date_str = info_paper_id.get("time", "0000-01-01 00:13:21+00:00")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S%z")
        year = date_obj.year
        paper_info["time"] = year
    return paper_info

# get the section of the paper
def get_section(key):
    if key == "Abstract":
        return abstract
    elif key == "Introduction":
        return introduction
    elif key == "Related Work":
        return related_work
    elif key == "Methodology":
        return methodology
    elif key == "Experiment":
        return experiment
    elif key == "Discussion":
        return discussion
    else:
        return "unknown"

# get the wikidata entity
def get_wikidata_entity(keyword):
    pre_result = wiki_dict.get(keyword, -1)
    if pre_result != -1:
        return pre_result[0], pre_result[1]
    else:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": keyword,
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if len(data["search"]) == 0:
                wiki_dict[keyword] = ("None", 0)
                return "None", 0
            wikidata_entity_id = data["search"][0]["id"]
            wikidata_entity_str = data["search"][0]["display"]["label"]["value"]

            wiki_dict[keyword] = (wikidata_entity_str, wikidata_entity_id)

            return wikidata_entity_str, wikidata_entity_id
        else:
            print(f"Error {response.status_code}: Failed to fetch data from Wikidata API")
            return "None", 0

#this function is used to get the location of the section in the paper
def get_paper_section_loc_char(paper_id):
    result = {}

    if mode == "ssh":
        splited_paper_path = "/home/users/u000000/askg/a-scholarly-kg/src/Papers/ASKG_Paper_Dataset/splitted_papers"
    else:
        splited_paper_path = "../ASKG_Paper_Dataset/splitted_papers"

    doc_path = os.path.join(splited_paper_path, paper_id + ".json")
    with open(doc_path, "r") as f:
        json_data = json.load(f)

        cur_len = 0
        for key in json_data.keys():
            cur_len = cur_len + len(json_data[key])
            result[key] = cur_len
    return result


def find_located_sec_TNNT(pos_idx_from, paper_section_loc_char):
    for key in paper_section_loc_char.keys():
        if pos_idx_from < paper_section_loc_char[key]:
            return key


def add_tnnt_ner_result(ASKG, paper_id, input_path_tnnt_result, section_dict, paper_info):
    paper_section_loc_char = get_paper_section_loc_char(paper_id)
    paper_title = paper_info["title"]

    if paper_id in tnnt_list:
        original_path = os.path.join(input_path_tnnt_result, paper_id + ".pdf-MEL+NER_output.json")
        with open(original_path, "r") as f:
            json_data = json.load(f)
            NER_result = json_data.get("NLP-NER", "")
            NER_result = NER_result[0].get('doc-0', "").get('spacy_lg_model', "").get("_output", "")
            if NER_result != "":
                for key in NER_result.keys():
                    for item in NER_result.get(key):
                        entity_str = item.get("entity")
                        wikidata_str, wikidata_ID = get_wikidata_entity(entity_str)
                        pos_idx_from = item.get("start_index")
                        pos_idx_to = item.get("end_index")
                        sentence = item.get("sentence")
                        if wikidata_ID == 0:
                            entity_str = quote(entity_str.replace(" ", "_"), safe='')
                        else:
                            entity_str = quote(entity_str.replace(" ", "_"),
                                                        safe='') + "-" + wikidata_ID
                        entity = ASKG_namespace_data[f"{key}-{entity_str}"]

                        Wikidata_entity = WIKIDATA_namespace[f"{wikidata_ID}"]
                        if (Wikidata_entity, None, None) not in ASKG and wikidata_ID != 0:
                            ltr_wikidata_name = Literal(wikidata_str, lang="en")
                            ASKG.add((Wikidata_entity, RDFS.label, ltr_wikidata_name))
                        if (entity, None, None) not in ASKG:
                            ltr_tnnt_entity_name = Literal(entity_str.lower(), datatype=XSD.string)
                            ASKG.add((entity, RDFS.label, ltr_tnnt_entity_name))
                            ASKG.add((entity, RDF.type, tnnt_dict[key]))
                            if (entity, OWL.sameAs, Wikidata_entity) not in ASKG and wikidata_ID != 0:
                                ASKG.add((entity, OWL.sameAs, Wikidata_entity))

                        loc_sec = find_located_sec_TNNT(pos_idx_from, paper_section_loc_char)
                        sec = section_dict[loc_sec]

                        excerpt_label_str = "Paper-" + f"{[paper_title]}" + " | " + "Section-" + f"{[loc_sec]}" + " | " + "Excerpt-" + f"{[pos_idx_from]}" + "-" + f"{[pos_idx_to]}"
                        excerpt_label = Literal(excerpt_label_str, lang="en")
                        hashed_excerpt = blake2s_hash(excerpt_label_str)
                        excerpt_entity = ASKG_namespace_data[f"Excerpt-{hashed_excerpt}"]

                        ASKG.add((sec, rel_contains, excerpt_entity))
                        ASKG.add((excerpt_entity, rel_mentions, entity))
                        ASKG.add((excerpt_entity, RDFS.label, excerpt_label))

                        ltr_pos_idx_from = Literal(pos_idx_from, datatype=XSD.int)
                        ltr_pos_idx_to = Literal(pos_idx_to, datatype=XSD.int)
                        ltr_sentence = Literal(sentence, datatype=XSD.string)
                        ASKG.add((excerpt_entity, rel_CharIndexFrom, ltr_pos_idx_from))
                        ASKG.add((excerpt_entity, rel_CharIndexTo, ltr_pos_idx_to))
                        ASKG.add((excerpt_entity, rel_in_sent, ltr_sentence))

    return ASKG

def construct_KG_Paper_Section(ASKG, paper_id, paper_info, json_data):

    global rel_summary
    rel_summary = ASKG_namespace_onto.summary

    #ASKG
    rel_id = ASKG_namespace_onto.id

    paper_title = paper_info["title"]
    first_author = paper_info["authors"][0]
    year = paper_info["time"]
    paper_label_str = f"[{paper_title}]" + "-" + f"[{first_author}]" + "-" + f"[{year}]"
    paper_hashed = blake2s_hash(paper_label_str)
    paper_entity = ASKG_namespace_data[f"Paper-{paper_hashed}"]

    if (paper_entity, None, None) in ASKG:
        return ASKG

    ASKG.add((paper_entity, RDF.type, paper))


    paper_label = Literal(paper_label_str, lang="en")
    ASKG.add((paper_entity, RDFS.label, paper_label))

    ltr_paper_id = Literal(paper_id, datatype=XSD.string)
    ASKG.add((paper_entity, rel_id, ltr_paper_id))

    ltr_paper_title = Literal(paper_info["title"], datatype=XSD.string)
    ASKG.add((paper_entity, DC.title, ltr_paper_title))

    #add link
    rel_link =  ASKG_namespace_onto.paperLink
    for link in paper_info["link"]:
        ltr_paper_link = Literal(link, datatype=XSD.string)
        ASKG.add((paper_entity, rel_link, ltr_paper_link))

    rel_DOI = ASKG_namespace_onto.doi
    if paper_info["doi"] != "":
        ltr_paper_doi = Literal(paper_info["doi"], datatype=XSD.string)
        ASKG.add((paper_entity, rel_DOI, ltr_paper_doi))

    section_dict = {}
    for key in json_data["sections"].keys():
        # add section label
        section_label = "Paper-" + f"[{paper_title}]" + "-" + f"[{first_author}]" + "-" + f"[{year}]" + " | " + "Section" + "-" + f"[{key}]"
        hashed_section = blake2s_hash(section_label)
        key_striped = key.replace(" ", "")
        sec = ASKG_namespace_data[f"{key_striped}-{hashed_section}"]
        section_dict[key] = sec
        ASKG.add((sec, RDF.type, get_section(key)))
        ASKG.add((paper_entity, rel_hasSection, sec))

        # add section label
        section_label = "Paper-" + f"[{paper_title}]" + "-" + f"[{first_author}]" + "-" +  f"[{year}]" + " | " + "Section" + "-" + f"[{key}]"
        ASKG.add((sec, RDFS.label, Literal(section_label, lang="en")))

        # add keywords part
        rel_similarity_score = ASKG_namespace_onto.similarityScore
        keywords = json_data["key_words"][key]

        for item in keywords:
            keyword = item[0]
            similarity_score = Literal(item[1], datatype=XSD.float)
            keywordOfSection_entity_str = section_label + keyword
            keywordOfSection_entity_hashed = blake2s_hash(keywordOfSection_entity_str)
            keywordOfSection_entity = ASKG_namespace_data[f"KeywordOfSection-{keywordOfSection_entity_hashed}"]
            ASKG.add((keywordOfSection_entity, RDF.type, KeywordOfSection))
            ASKG.add((keywordOfSection_entity, rel_similarity_score, similarity_score))
            ASKG.add((sec, rel_domo_keyword, keywordOfSection_entity))
            wikidata_entity_str, wikidata_entity_ID = get_wikidata_entity(keyword)
            if wikidata_entity_ID != 0:
                keyword_str = quote(keyword.lower().replace(" ", "_"), safe='')  + "-" + str(wikidata_entity_ID)
            else:
                keyword_str = quote(keyword.lower().replace(" ", "_"), safe='')
            # keyword_hashed = blake2s_hash(keyword_str)
            keyword_entity = ASKG_namespace_data[f"Keyword-{keyword_str}"]
            wikidata_entity = WIKIDATA_namespace[f"{wikidata_entity_ID}"]
            if (keyword_entity, None, None) not in ASKG:
                ASKG.add((keyword_entity, RDF.type, Keyword))
                ASKG.add((keyword_entity, RDFS.label, Literal(keyword, lang="en")))
            if wikidata_entity_ID != 0:
                if (wikidata_entity, None, None) not in ASKG:
                    ASKG.add((wikidata_entity, RDFS.label, Literal(wikidata_entity_str, lang="en")))
                ASKG.add((keyword_entity, OWL.sameAs, wikidata_entity))
            ASKG.add((keywordOfSection_entity, rel_correspondsTo, keyword_entity))


        # add summary part
        summarization = json_data["summarization"][key]
        summarization = Literal(summarization, datatype=XSD.string)
        ASKG.add((sec, rel_summary, summarization))

    if mode == "local":
        input_path_excerpt = "../ASKG_Paper_Dataset/AcademicEntity"
    else:
        input_path_excerpt = "/home/users/u000000/askg/a-scholarly-kg/src/Papers/ASKG_Paper_Dataset/AcademicEntity"

    #add excerpt part to ASKG
    if paper_id == "1607.08822":
        excerpt_file_path = os.path.join(input_path_excerpt, paper_id + ".json")
        if os.path.isfile(excerpt_file_path):
            with open(excerpt_file_path, "r") as excerpt_file:
                excerpt_result = json.load(excerpt_file)
            ASKG = construct_KG_Excerpt_Academic_Entity(ASKG, excerpt_result, paper_id, section_dict, paper_info)

    if mode == "ssh":
        input_path_tnnt_result = "/home/users/u000000/askg/a-scholarly-kg/src/Papers/ASKG_Paper_Dataset/MEL_TNNT_NER_Result"
    else:
        input_path_tnnt_result = "../ASKG_Paper_Dataset/MEL_TNNT_NER_Result"


    #add TNNT part to ASKG
    ASKG = add_tnnt_ner_result(ASKG, paper_id, input_path_tnnt_result, section_dict, paper_info)

    return ASKG

def create_tnnt_list(input_path_tnnt_result):

    items_in_directory = os.listdir(input_path_tnnt_result)

    for item in items_in_directory:
        item_path = os.path.join(input_path_tnnt_result, item)

        if os.path.isfile(item_path):
            pattern = r"^(.+)\.pdf"
            match1 = re.match(pattern, item)

            if match1:
                tnnt_list.append(match1.group(1))

if __name__ == "__main__":
    if mode == "ssh":
        input_path_paper_info = "/home/users/u000000/askg/a-scholarly-kg/src/Papers/ASKG_Paper_Dataset/paper_info/papers_info.json"
        input_path_paper = "/home/users/u000000/askg/a-scholarly-kg/src/Papers/ASKG_Paper_Dataset/summarization"
    else:
        input_path_paper_info = "../ASKG_Paper_Dataset/paper_info/papers_info.json"
        input_path_paper = "../ASKG_Paper_Dataset/summarization"

    if mode == "ssh":
        input_path_tnnt_result = "/home/users/u000000/askg/a-scholarly-kg/src/Papers/ASKG_Paper_Dataset/MEL_TNNT_NER_Result"
    else:
        input_path_tnnt_result = "../ASKG_Paper_Dataset/MEL_TNNT_NER_Result"

    create_tnnt_list(input_path_tnnt_result)

    with open(input_path_paper_info, "r") as f_info:
        json_paper_info = json.load(f_info)

    ASKG = construct_KG_Paper_Abs(ASKG)

    cnt = 0
    for root, dirs, files in os.walk(input_path_paper):
        total_files = len(files)
        for idx, filename in enumerate(files):
            file_path = os.path.join(input_path_paper, filename)

            with open(file_path, "r") as f:
                json_data = json.load(f)

            paper_id = re.sub(r"\.json$", "", filename)
            paper_info = get_paper_info(json_paper_info, paper_id)

            ASKG = construct_KG_Paper_Section(ASKG, paper_id, paper_info, json_data)

            rdf_ttl_data = ASKG.serialize(format='ttl') #ttl
            if mode == "ssh":
                output_path = f"/home/users/u000000/askg/anu-scholarly-kg/src/Papers/RDF/ASKG_{cnt}.ttl"
                wiki_dict_output_path = f"/home/users/u000000/askg/anu-scholarly-kg/src/Papers/RDF/ASKG_{cnt}_wiki_dict.json"

                with open(output_path, "wb") as f:
                    f.write(rdf_ttl_data.encode("utf-8"))
                print(cnt)

                with open(wiki_dict_output_path, "w") as f:
                    json.dump(wiki_dict, f)
            else:
                print(cnt)
                output_path = f"ASKG_local_{cnt}.ttl"
                wiki_dict_output_path = f"ASKG_local_wiki_dict_{cnt}.json"

                with open(output_path, "wb") as f:
                    f.write(rdf_ttl_data.encode("utf-8"))

                # with open(wiki_dict_output_path, "w") as f:
                #     json.dump(wiki_dict, f)

            cnt += 1


