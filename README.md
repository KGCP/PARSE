![Pipeline_Small](docs/pipeline_small.png)

___

# Papers And Relationships Semantic Extraction (PARSE)

This repository contains the code for our research paper titled "ASKG: An Approach to Enrich Scholarly Knowledge
Graphs through Paper Decomposition with Deep
Learning". Our project's primary focus is to expand the academic knowledge graph automatically by extracting fine-grained knowledge from academic papers, particularly those related to computer science.

---

## Architecture Overview

![Pipeline](docs/overall_pipeline.png)

The components is divided into two main parts:

1. **Extracting General and Specific Information**: Firstly, an scholarly papers database needs to be established for the following knowledge extraction process. Meanwhile, a JSON file describing the metadata of each paper in the paper’s database is required to be generated when the database was initially established. This JSON file should contain each paper’s metadata, including the title, authors, and category, etc. In our proposed pipeline, the categories that papers belong to was classified according to arXiv. The automatic pipeline could be generally divided into 2 main parts. One part is for extracting general information from academic papers. In this part, All the research papers are ported into the well-established MEL & TNNT system to extract the metadata and general name entities, such as person name, nationalities and political groups, buildings, and bridges names, etc. The output metadata and general NER results will be processed to expand the existing scholarly knowledge graph. 

2. **Extracting Domain-specific Knowledge**: The other part is to extract the specific scientific fields’ knowledge from academic papers. Currently, our research project mainly focuses on computer science-related papers. Therefore, firstly the JSON file describing the metadata of each paper in the paper’s database was input to the paper statistical analyzer to obtain the metadata about the whole database and identify all the papers related to the computer science field and generate a target papers list. Then the pipeline will send HTTP requests to TNNT servers to obtain the raw text of papers according to the target papers list. On one hand, the raw texts of papers are then sent to the NER module which is built with several pre-trained transformer-based large-scale language models to obtain computer science-related academic terms. Then the academic terms are compared and integrated with academic terms from Wikidata and other knowledge graph databases for name entity linking. On the other hand, the raw texts of papers are sent to the automatic summarization model BRIO and keyword model KeyBERT to generate summaries and recognize keywords, further enriching the existing academic knowledge graph. 

## Core Features

### Name Entity Recognition Module

![Pipeline](docs/NER_model.jpeg)
Figure shows the structure of our proposed NER module. The NER model can be divided into 4 main parts. The first part is the input layer for cleaning the raw academic paper text acquired from the MEL & TNNT system and feature engineering. The second is the word embedding part. All the words are tokenized and embedded with pre-trained large language models in this part. Then in the third part, the word embeddings are input into the BiLSTM layer for feature learning. Finally, in the last part of this model, the output layer, the linear CRF layer will calculate the tag score and map the predefined tag. 

The output of the NER module is a JSON format file. In this NER model, all identified entities are classified into 7 scientific types shown in the following tables.

| Types             | Definition |
| ----------------- | ---------- |
| research problem  | theme of the investigation |
| method            | existing protocols used to support the solution |
| solution          | novel contribution of a work that solves the research problem |
| tool              | found by asking the question “Using what?” or “By which means?” |
| resource          | names of existing data and other references to utilities like the Web, Encyclopedia, etc., used to address the research problem or used in the solution |
| dataset           | name of a dataset |
| language          | natural language focus of a work |

The sample output is shown below:
```json
{
  "RESEARCH_PROBLEM": 
    "computational linguistics": {
        "position": [
        [1713, 1714]
      ],
      "sentence": [
        "transforming a sentence into its meaning representation also has received considerable attention within the computational linguistics community."
      ]
    }
}
```
The following is enriched ontology after NER module:
![Pipeline](docs/ASKG_NER.png)

### Name Entity Linking Module
Name entity linking is an important task in knowledge graph construction. NEL mainly focuses on linking the identified name entities to their corresponding entries in the knowledge base. NEL can help improve the relevance and context relationship between scholarly knowledge graphs and make keyword search and navigation possible.

NEL also contributes to enhancing storage efficiency and computational speed. By applying NEL, duplication of entities and triples can be notably reduced. The model achieves this by identifying and merging entities that are identical or highly similar. This process not only conserves storage space but also optimizes the knowledge graph computing speed.

The NEL module output is like the following:
```json
{
  "RESEARCH_PROBLEM": {
    "computational linguistics": {
      "position": [
        [1713,1714]
      ],
      "sentence": [
        "transforming a sentence into its meaning representation also has received considerable attention within the computational linguistics community"
      ],
      "is_matched": true,
      "exact_match": true,
      "wikidata_ID": "Q182557",
      "wikidata_desc": "interdisciplinary field",
      "matched_words": "computational linguistics",
      "cosine similarity": 1.0
    }
  }
}
```
The enriched ontology after NEL module is shown below:
![Pipeline](docs/ASKG_NEL.png)

### Automatic Summarization and Keyword Extraction Module
In our pipeline, we integrate BRIO model to conduct the text summarization work. we also integrate the KeyBERT model into our pipeline.

The enriched ontology after automatic summarization and keyword extraction module is shown below:
![Pipeline](docs/ASKG_Summary.png)

### MEL & TNNT Module
MEL \& TNNT, when combined, provide us with a powerful toolset that can extract textual content from various file formats and identify named entities using natural language processing technologies. 
The enriched ontology after MEL & TNNT module is shown below:
![Pipeline](docs/ASKG_TNNT.png)

## Enriched ASKG Ontology
![Pipeline](docs/ASKG_ALL.png)

## Evaluation of Overall Knowledge Graph
In the final step, we integrate the results of the generic entities identified by MEL & TNNT, along with the previous NER, NEL, automatic summarization, and keyword extraction output, into the ASKG. In this way, we have obtained the fully enriched ASKG.

After the final enrichment of the ASKG, the evolution of ASKG is as follows: 

| Metrics                      | Original ASKG | Enriched ASKG | Change (%) |
| ---------------------------- | ------------- | ------------- | ---------- |
| Number of relation types    | 21            | 40            | +90.48     |
| Number of entity types      | 18            | 46            | +155.56    |
| Number of entities          | 235,314       | 1,215,106     | +416.38    |
| Number of triples           | 1,048,576     | 2,866,980     | +173.42    |
| Average degree              | 4.46          | 4.72          | +5.83      |
| Clustering coefficient     | 6.28e-05      | 1.86e-04      | +196.18    |
| Number of connected components | 6         | 1             | -83.33     |
| Information density         | 4.46          | 2.36          | -47.09     |

Table: Comparison of the original ASKG and overall enriched ASKG


The number of relationship types increases significantly from 21 to 40, representing the wider variety of interactions captured between entities. Similarly, the number of entity types sees a substantial jump from 18 to 46, indicating the incorporation of a greater diversity of entities. 

In terms of the number of entities, there is a considerable increase from 235,314 to 1,215,106, showing the successful integration of entities from NER, NEL, automatic summarization, keyword extraction, and MEL & TNNT output. 

The number of triples shows a similar pattern, expanding from 1,048,576 to 2,866,980. This growth is indicative of the successful inclusion of academic knowledge and facts into the ASKG. 

Despite the above improvement, the average degree increased from 4.46 to 4.72, representing a slight increase in the connections between entities. The clustering coefficient also increases from 6.28e-05 to 1.86e-04, showing a tighter interconnection of the ASKG. 

The number of connected components decreases from 6 to 1, ensuring the robust connectivity of the ASKG. This confirms the ASKG’s integrity and its utility for downstream applications. 

Lastly, the information density decreases from 4.46 to 2.36. This drop is due to the significant growth in the number of entities and triples, which outpaces the growth in information intensity. The decrease in information density can be seen as a tradeoff for the enrichment of the ASKG, with a larger and more diverse set of knowledge now represented.

## Knowledge Graph Visualization
![Pipeline](docs/ASKG_neo4j.png)
To do this, we use the neosemantics (n10s) provided by Neo4j to convert and import the RDF graph into the Neo4j database and run relevant Cypher queries. In this process, we delete existing nodes and relationships, and create a unique constraint to ensure that the resource’s URI is unique.