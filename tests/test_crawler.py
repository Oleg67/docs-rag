import httpx
import pymupdf
import pytest

from docs_rag import crawler
from docs_rag.config import CrawlerSettings, WebSource
from docs_rag.crawler import WebCrawler


def _pdf() -> bytes:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Linked PDF text.")
    return pdf.tobytes()


SITE = {
    "/robots.txt": ("text/plain", "User-agent: *\nDisallow: /guide/private\n"),
    "/guide/": ("text/html", '<h1>Home</h1><p>Start.</p><a href="a.html">A</a>'
                '<a href="private.html">P</a><a href="/blog/x.html">Blog</a>'
                '<a href="https://other.org/guide/">Other</a><a href="/files/manual.pdf">PDF</a>'),
    "/guide/a.html": ("text/html", '<p>Page A.</p><a href="/guide/">Home</a>'
                      '<a href="/guide/redirect">R</a>'),
    "/guide/redirect": ("redirect", "https://other.org/stolen"),
    "/guide/private.html": ("text/html", "<p>Secret.</p>"),
    "/blog/x.html": ("text/html", "<p>Out of prefix.</p>"),
    "/files/manual.pdf": ("application/pdf", _pdf()),
}


def _handler(site: dict, requested: list[str]):
    def handle(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host != "docs.example.org" or request.url.path not in site:
            return httpx.Response(404)
        kind, body = site[request.url.path]
        if kind == "redirect":
            return httpx.Response(302, headers={"location": body})
        if isinstance(body, int):
            return httpx.Response(body)
        return httpx.Response(200, headers={"content-type": kind}, content=body)

    return handle


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(crawler.time, "sleep", delays.append)
    return delays


def _crawl(site: dict, requested: list[str], **source) -> list:
    web = WebSource(type="web", name="docs", url="https://docs.example.org/guide/", **source)
    transport = httpx.MockTransport(_handler(site, requested))
    return WebCrawler(web, CrawlerSettings(), transport=transport).crawl()


def test_crawl_stays_in_scope_and_respects_robots(no_sleep: list[float]) -> None:
    requested: list[str] = []
    documents = _crawl(SITE, requested)

    assert {d.source for d in documents} == {
        "https://docs.example.org/guide/",
        "https://docs.example.org/guide/a.html",
        "https://docs.example.org/files/manual.pdf",
    }
    pdf = next(d for d in documents if d.kind == "pdf")
    assert pdf.sections[0].page == 1 and pdf.title == "manual.pdf"
    assert not any("other.org" in url or "private" in url or "blog" in url for url in requested)
    assert no_sleep and min(no_sleep) >= 1.0


def test_max_pages_and_pdfs_off() -> None:
    documents = _crawl(SITE, [], max_pages=1, pdfs=False)
    assert [d.source for d in documents] == ["https://docs.example.org/guide/"]


def test_unreachable_robots_txt_stops_the_crawl() -> None:
    site = {**SITE, "/robots.txt": ("text/plain", 503)}
    with pytest.raises(RuntimeError, match="robots.txt"):
        _crawl(site, [])


def test_crawl_delay_from_robots_is_honoured(no_sleep: list[float]) -> None:
    site = {**SITE, "/robots.txt": ("text/plain", "User-agent: *\nCrawl-delay: 3\n")}
    _crawl(site, [], max_pages=1)
    assert set(no_sleep) == {3}


def test_many_server_errors_refuse_a_partial_crawl() -> None:
    site = {**SITE, "/guide/a.html": ("text/html", 500)}
    with pytest.raises(RuntimeError, match="partial crawl"):
        _crawl(site, [])
