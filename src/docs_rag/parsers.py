"""Turn HTML, Markdown, plain text and PDF into sections of blank-line-separated paragraphs."""

import re
from urllib.parse import urldefrag, urljoin

import pymupdf
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from .schemas import Section

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCKS = {
    "p", "div", "li", "ul", "ol", "pre", "blockquote", "table", "tr", "td", "th", "dl", "dt", "dd",
    "figure", "figcaption", "section", "article", "main",
}
# Page chrome, not content. Links are collected before these are removed, so menus still guide
# the crawler.
_CHROME = ["script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside",
           "form", "iframe"]
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _clean(text: str) -> str:
    return " ".join(text.split())


def _paragraph_sections(items: list[tuple[str, str]]) -> list[Section]:
    """Group ("heading" | "paragraph", text) items into sections that start at each heading."""
    sections: list[Section] = []
    heading: str | None = None
    paragraphs: list[str] = []

    def flush() -> None:
        if paragraphs:
            sections.append(Section(heading=heading, text="\n\n".join(paragraphs)))

    for kind, text in items:
        if kind == "heading":
            flush()
            heading, paragraphs = text, []
        else:
            paragraphs.append(text)
    flush()
    return sections


def parse_html(
    html: str, base_url: str | None = None
) -> tuple[str | None, list[Section], list[str]]:
    """Return the page title, its content sections and (with base_url) the absolute links."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    if base_url:
        for anchor in soup.find_all("a", href=True):
            links.append(urldefrag(urljoin(base_url, anchor["href"])).url)
    title_node = soup.find("meta", property="og:title")
    if title_node and title_node.get("content", "").strip():
        title = _clean(title_node["content"])
    else:
        node = soup.find("title") or soup.find("h1")
        title = _clean(node.get_text(" ")) if node else None
    for tag in soup(_CHROME):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup

    items: list[tuple[str, str]] = []
    parts: list[str] = []

    def flush() -> None:
        text = _clean("".join(parts))
        if text:
            items.append(("paragraph", text))
        parts.clear()

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif child.name in _HEADINGS:
                flush()
                if heading := _clean(child.get_text(" ")):
                    items.append(("heading", heading))
            elif child.name == "br":
                flush()
            elif child.name in _BLOCKS:
                flush()
                walk(child)
                flush()
            else:
                walk(child)

    walk(root)
    flush()
    return title, _paragraph_sections(items), links


def parse_markdown(text: str) -> tuple[str | None, list[Section]]:
    """Split on ATX headings (`#`, `##`, ...); the first `# ` heading is the title."""
    items: list[tuple[str, str]] = []
    title: str | None = None
    lines: list[str] = []
    in_code = False

    def flush() -> None:
        if paragraph := "\n".join(lines).strip():
            items.append(("paragraph", paragraph))
        lines.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _MARKDOWN_HEADING.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
            if match.group(1) == "#" and title is None:
                title = heading
            items.append(("heading", heading))
        elif not line.strip() and not in_code:
            flush()
        else:
            lines.append(line)
    flush()
    return title, _paragraph_sections(items)


def parse_text(text: str) -> list[Section]:
    paragraphs = [_clean(part) for part in re.split(r"\n\s*\n", text)]
    return _paragraph_sections([("paragraph", part) for part in paragraphs if part])


def parse_pdf(data: bytes) -> tuple[str | None, list[Section]]:
    """One section per page, so citations can give page numbers. Scanned PDFs yield no text."""
    sections: list[Section] = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        title = (pdf.metadata or {}).get("title") or None
        for number, page in enumerate(pdf, start=1):
            paragraphs = []
            for block in page.get_text("blocks", sort=True):
                if block[6] != 0:  # image block
                    continue
                # Re-join words hyphenated across lines, then unwrap the lines.
                text = _clean(re.sub(r"(\w)-\n(\w)", r"\1\2", block[4]))
                if text:
                    paragraphs.append(text)
            if paragraphs:
                sections.append(Section(page=number, text="\n\n".join(paragraphs)))
    return title, sections
