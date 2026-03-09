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
    assert chunks[0].text == "Intro paragraph."
    assert chunks[0].page_start == 0
    assert chunks[0].page_end == 0
    assert chunks[0].order_in_paper == 0

    assert chunks[1].chunk_id == "sample-paper_sec_1"
    assert chunks[1].section_type == "method"
    assert chunks[1].text == "Method paragraph one.\nMethod paragraph two."
    assert chunks[1].order_in_paper == 1
