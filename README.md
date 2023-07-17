# Knowledge Graph Construction Pipeline (KGCP)

This repository contains the code for our research paper titled "Automated Construction of Academic Knowledge Graph (ASKG) using KGCP". Our project's primary focus is to expand the academic knowledge graph automatically by extracting fine-grained knowledge from academic papers, particularly those related to computer science.

## Architecture Overview

The components is divided into two main parts:

1. **Extracting General and Specific Information**: We start by scraping web pages of MAKG, Wikidata, and ANU's schools to collect researcher information and their papers, thus building a rich academic paper dataset. This dataset is used to extract metadata (author names, affiliated institutions, contact information, grants, areas of expertise, etc.), paper content, and general named entities (like people, nationalities, locations, etc.). All the extracted data is used to enrich the existing ASKG.

2. **Extracting Domain-specific Knowledge**: Here, the focus is on computer science (CS) papers. We use a paper statistical analyser to identify all CS-related papers from the dataset, which then forms our target paper list. These papers are processed according to the IMRaD (Introduction, Methods, Results, and Discussion) format for identification and division. We use several Transformer-based NER models like RoBERTa, SciBERT, and LinkBERT, etc., to extract academic entities related to computer science.

## Project Components

* **Web Scraping**: We use web scraping technologies to collect researchers' information and their papers, thereby forming a rich dataset for knowledge extraction.

* **Metadata Extraction**: We automatically generate a JSON file that describes the metadata of each paper, including details such as the title, author, link, time, and classification.

* **NER Module**: The NER module uses Transformer-based models to extract academic entities related to computer science from the paper text.

* **CharCNN**: To address potential character loss or errors during the conversion of PDF-format academic papers to plain text, we use CharCNN as a character embedding layer to enhance model robustness.

* **Named Entity Linking**: Academic entities extracted from the NER module are categorized and compared with entities in Wikidata and other knowledge bases to link named entities and enrich our knowledge graph.

* **Automatic Summarization and Keyword Recognition**: We use the automatic summarization model BRIO and keyword model KeyBERT to generate summaries and recognize keywords, further enriching the existing academic knowledge graph.

## How to Use


## References


For more details about our work, please refer to our paper. We welcome any kind of feedback and contribution.
