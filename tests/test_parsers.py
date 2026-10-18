import pymupdf

from docs_rag.parsers import parse_html, parse_markdown, parse_pdf, parse_text

HTML = """<html><head><title>Guide</title></head><body>
<nav><a href="/guide/b.html">Next</a></nav>
<main><h1>Intro</h1><p>First <b>para</b>.</p><ul><li>One</li><li>Two</li></ul>
<h2>Usage</h2><p>Run it.<br>Then stop.</p><script>alert(1)</script>
<a href="c.html#top">C</a></main><footer>© 2026</footer></body></html>"""


def test_html_sections_skip_page_chrome_but_keep_menu_links() -> None:
    title, sections, links = parse_html(HTML, "https://x.org/guide/a.html")

    assert title == "Guide"
    assert [(s.heading, s.text) for s in sections] == [
        ("Intro", "First para.\n\nOne\n\nTwo"),
        ("Usage", "Run it.\n\nThen stop.\n\nC"),
    ]
    assert links == ["https://x.org/guide/b.html", "https://x.org/guide/c.html"]


def test_markdown_headings_split_sections_but_not_inside_code() -> None:
    title, sections = parse_markdown(
        "# Title\n\nIntro text\nwraps.\n\n## Setup\n\n```sh\n# not a heading\n\nls\n```\n"
    )

    assert title == "Title"
    assert [s.heading for s in sections] == ["Title", "Setup"]
    assert sections[0].text == "Intro text\nwraps."
    assert "# not a heading" in sections[1].text


def test_text_paragraphs() -> None:
    assert parse_text("a\nb\n\n\nc")[0].text == "a b\n\nc"


def test_pdf_has_one_section_per_page() -> None:
    pdf = pymupdf.open()
    for words in ("Lactate threshold rises with train-\ning.", "Second page."):
        pdf.new_page().insert_text((72, 72), words)
    pdf.set_metadata({"title": "Physiology"})

    title, sections = parse_pdf(pdf.tobytes())

    assert title == "Physiology"
    assert [s.page for s in sections] == [1, 2]
    assert sections[0].text == "Lactate threshold rises with training."
