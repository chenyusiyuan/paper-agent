from app.parsing.tei_parser import TeiParser


MINIMAL_TEI = """\
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Sample Paper</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename>Alice</forename>
                <surname>Smith</surname>
              </persName>
            </author>
            <author>
              <persName>
                <forename>Bob</forename>
                <surname>Lee</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <title>ACL</title>
            <imprint>
              <date when="2024-01-01" />
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>This paper studies retrieval agents.</p>
      </abstract>
      <textClass>
        <keywords>
          <term>retrieval</term>
          <term>agent</term>
        </keywords>
      </textClass>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>Intro paragraph.</p>
      </div>
      <div>
        <head>Methodology</head>
        <p>Method paragraph one.</p>
        <p>Method paragraph two.</p>
      </div>
    </body>
  </text>
</TEI>
"""

NESTED_TEI = """\
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Nested Paper</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename>Alice</forename>
                <surname>Smith</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <title>ACL</title>
            <imprint>
              <date when="2025-01-01" />
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>Nested parsing example.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>3 Methodology</head>
        <p>Overview paragraph.</p>
        <div>
          <head>3.1 Model</head>
          <p>Model details.</p>
          <div>
            <head>3.1.1 Inner</head>
            <p>Inner details should stay out of fine chunk.</p>
          </div>
        </div>
        <div>
          <head>3.2 Training</head>
          <p>Training details.</p>
        </div>
      </div>
      <div>
        <head>4 Conclusion</head>
        <p>Closing thoughts.</p>
      </div>
    </body>
  </text>
</TEI>
"""

MESSY_TEI = """\
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Messy Paper</title>
      </titleStmt>
      <publicationStmt>
        <date>2024</date>
        <publisher>Association for Computational Linguistics</publisher>
      </publicationStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename>Dario</forename>
                <surname>2020 Amodei</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <meeting>
              <title>EMNLP 2024</title>
            </meeting>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>Processing long contexts is hard. * Corresponding Author. † Equal contribution.</p>
      </abstract>
      <textClass>
        <keywords>Large Language Model • Recommender System; Retrieval-Augmented Generation</keywords>
      </textClass>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>3 Methodology</head>
        <p>Method overview.</p>
      </div>
      <div>
        <head>Joe Biden announced his candidacy for the 2020 presidential election on</head>
        <p>Bad extracted heading should fall back.</p>
      </div>
      <div>
        <head># Architecture</head>
        <p>Architecture details.</p>
      </div>
      <div>
        <head>Section 12</head>
        <p>Generic placeholder heading should fall back.</p>
      </div>
    </body>
  </text>
</TEI>
"""


def test_parse_returns_metadata_and_chunks() -> None:
    parser = TeiParser()

    metadata, chunks = parser.parse(MINIMAL_TEI, "sample-paper")

    assert metadata.paper_id == "sample-paper"
    assert metadata.title == "Sample Paper"
    assert metadata.authors == ["Alice Smith", "Bob Lee"]
    assert metadata.year == 2024
    assert metadata.venue == "ACL"
    assert metadata.abstract == "This paper studies retrieval agents."
    assert metadata.keywords == ["retrieval", "agent"]
    assert metadata.section_titles == ["Introduction", "Methodology"]

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "sample-paper_sec_0"
    assert chunks[0].section_type == "introduction"
    assert chunks[0].section_title == "Introduction"
    assert chunks[0].section_path == "Introduction"
    assert chunks[0].text == "Intro paragraph."
    assert chunks[0].page_start == 0
    assert chunks[0].page_end == 0
    assert chunks[0].order_in_paper == 0
    assert chunks[0].level == 0
    assert chunks[0].parent_chunk_id is None
    assert chunks[0].granularity == "coarse"

    assert chunks[1].chunk_id == "sample-paper_sec_1"
    assert chunks[1].section_type == "method"
    assert chunks[1].text == "Method paragraph one.\nMethod paragraph two."
    assert chunks[1].order_in_paper == 1
    assert chunks[1].level == 0
    assert chunks[1].parent_chunk_id is None
    assert chunks[1].granularity == "coarse"


def test_parser_generates_coarse_and_fine_chunks() -> None:
    parser = TeiParser()

    _, chunks = parser.parse(NESTED_TEI, "nested-paper")

    assert len(chunks) == 5

    methodology_coarse = chunks[0]
    model_fine = chunks[1]
    inner_fine = chunks[2]
    training_fine = chunks[3]
    conclusion_coarse = chunks[4]

    assert methodology_coarse.granularity == "coarse"
    assert methodology_coarse.level == 0
    assert methodology_coarse.parent_chunk_id is None
    assert methodology_coarse.text == (
        "Overview paragraph.\n"
        "Model details.\n"
        "Inner details should stay out of fine chunk.\n"
        "Training details."
    )

    assert model_fine.granularity == "fine"
    assert model_fine.level == 1
    assert model_fine.parent_chunk_id == methodology_coarse.chunk_id
    assert model_fine.text == "Model details."

    assert inner_fine.granularity == "fine"
    assert inner_fine.level == 2
    assert inner_fine.parent_chunk_id == model_fine.chunk_id
    assert inner_fine.text == "Inner details should stay out of fine chunk."

    assert training_fine.granularity == "fine"
    assert training_fine.level == 1
    assert training_fine.parent_chunk_id == methodology_coarse.chunk_id
    assert training_fine.text == "Training details."

    assert conclusion_coarse.granularity == "coarse"
    assert conclusion_coarse.level == 0
    assert conclusion_coarse.parent_chunk_id is None


def test_parser_section_path() -> None:
    parser = TeiParser()

    _, chunks = parser.parse(NESTED_TEI, "nested-paper")

    assert chunks[0].section_path == "Methodology"
    assert chunks[1].section_path == "Methodology > Model"
    assert chunks[2].section_path == "Methodology > Model > Inner"


def test_parser_order_in_paper_is_global() -> None:
    parser = TeiParser()

    _, chunks = parser.parse(NESTED_TEI, "nested-paper")

    orders = [chunk.order_in_paper for chunk in chunks]
    assert orders == [0, 1, 2, 3, 4]
    assert len(orders) == len(set(orders))


def test_parser_cleans_metadata_and_bad_section_titles() -> None:
    parser = TeiParser()

    metadata, chunks = parser.parse(MESSY_TEI, "messy-paper")

    assert metadata.authors == ["Dario Amodei"]
    assert metadata.year == 2024
    assert metadata.venue == "EMNLP 2024"
    assert metadata.abstract == "Processing long contexts is hard."
    assert metadata.keywords == [
        "Large Language Model",
        "Recommender System",
        "Retrieval-Augmented Generation",
    ]
    assert metadata.section_titles == ["Methodology", "Section 2", "Architecture", "Section 4"]

    assert chunks[0].section_title == "Methodology"
    assert chunks[1].section_title == "Section 2"
    assert chunks[2].section_title == "Architecture"
    assert chunks[3].section_title == "Section 4"
