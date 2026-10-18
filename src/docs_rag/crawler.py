"""Polite same-host web crawler: robots.txt, a path prefix, at most one request per second."""

import logging
import time
from collections import deque
from datetime import UTC, datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from .config import CrawlerSettings, WebSource
from .parsers import parse_html, parse_pdf
from .schemas import Document

logger = logging.getLogger(__name__)


class WebCrawler:
    """Breadth-first crawl from a start URL, limited to its host and path prefix."""

    def __init__(
        self,
        source: WebSource,
        settings: CrawlerSettings,
        transport: httpx.BaseTransport | None = None,  # tests pass an httpx.MockTransport
    ) -> None:
        self.source = source
        self.settings = settings
        start = urlparse(str(source.url))
        self.host = start.netloc.lower()
        self.scheme = start.scheme
        self.prefix = source.path_prefix or start.path[: start.path.rfind("/") + 1] or "/"
        self._failed: list[str] = []
        self._client = httpx.Client(
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
            timeout=settings.timeout_seconds,
            transport=transport,
            event_hooks={"request": [self._check_request]},
        )
        self._robots = RobotFileParser()
        self._robots.allow_all = True  # only while _load_robots follows robots.txt redirects
        self._robots = self._load_robots()
        # Honour a slower Crawl-delay from robots.txt, never a faster one.
        self._delay = max(
            settings.delay_seconds, self._robots.crawl_delay(settings.user_agent) or 0
        )

    def _load_robots(self) -> RobotFileParser:
        """RFC 9309: 401/403 disallow everything, other 4xx allow everything, and 5xx or a
        network error stop the crawl (fail closed)."""
        robots_url = f"{self.scheme}://{self.host}/robots.txt"
        parser = RobotFileParser(robots_url)
        try:
            response = self._client.get(robots_url)
        except httpx.HTTPError as error:
            raise RuntimeError(f"Could not fetch {robots_url}: {error}") from error
        if response.status_code in {401, 403}:
            parser.disallow_all = True
        elif 400 <= response.status_code < 500:
            parser.allow_all = True
        elif response.is_success:
            parser.parse(response.text.splitlines())
        else:
            raise RuntimeError(f"Could not fetch {robots_url}: HTTP {response.status_code}")
        return parser

    def normalise(self, url: str) -> str | None:
        """The canonical form of an in-scope URL, or None for one the crawler must not fetch."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != self.host:
            return None
        path = parsed.path or "/"
        in_prefix = path.startswith(self.prefix) or path + "/" == self.prefix
        # Linked PDFs often live elsewhere on the host (/files/...), so they may leave the prefix.
        if not in_prefix and not (self.source.pdfs and path.lower().endswith(".pdf")):
            return None
        return parsed._replace(netloc=self.host, fragment="").geturl()

    def _allowed(self, url: str) -> bool:
        return self.normalise(url) is not None and self._robots.can_fetch(
            self.settings.user_agent, url
        )

    def _check_request(self, request: httpx.Request) -> None:
        """Runs before every request, including redirects, so they cannot leave the scope."""
        url = str(request.url)
        if url.endswith("/robots.txt") and urlparse(url).netloc.lower() == self.host:
            return
        if not self._allowed(url):
            raise httpx.RequestError(f"Refusing to fetch {url}", request=request)

    def _get(self, url: str) -> httpx.Response | None:
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as error:
            logger.warning("Skipping %s: %s", url, error)
            # Outages and rate limiting count as failures; refused URLs and 404s do not.
            if isinstance(error, httpx.TransportError) or (
                isinstance(error, httpx.HTTPStatusError)
                and (error.response.status_code >= 500 or error.response.status_code == 429)
            ):
                self._failed.append(url)
            return None
        finally:
            time.sleep(self._delay)  # after every request, including failed ones
        return response

    def crawl(self) -> list[Document]:
        start = self.normalise(str(self.source.url))
        if start is None:
            raise ValueError(f"{self.source.url} is outside its own path prefix {self.prefix!r}")
        queue, seen = deque([start]), {start}
        documents: list[Document] = []
        fetched = 0
        try:
            while queue and fetched < self.source.max_pages:
                url = queue.popleft()
                if not self._robots.can_fetch(self.settings.user_agent, url):
                    continue
                fetched += 1
                response = self._get(url)
                if response is None:
                    continue
                parsed = self._document(str(response.url), response)
                if parsed is None:
                    continue
                document, links = parsed
                if document.sections:
                    documents.append(document)
                for link in links:
                    if (link := self.normalise(link)) and link not in seen:
                        seen.add(link)
                        queue.append(link)
        finally:
            self._client.close()
        if fetched and len(self._failed) > self.settings.max_failed_fraction * fetched:
            raise RuntimeError(
                f"{len(self._failed)} of {fetched} requests to {self.host} failed (outage or rate "
                "limiting); not saving a partial crawl. Try again later."
            )
        return documents

    def _document(self, url: str, response: httpx.Response) -> tuple[Document, list[str]] | None:
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        now = datetime.now(UTC)
        if content_type in {"text/html", "application/xhtml+xml"}:
            title, sections, links = parse_html(response.text, url)
            document = Document(
                source=url, origin=self.source.name, kind="web", title=title or url,
                sections=sections, fetched_at=now,
            )
            return document, links
        if content_type == "application/pdf" and self.source.pdfs:
            if len(response.content) > self.settings.max_pdf_bytes:
                logger.warning("Skipping %s: PDF larger than %d bytes", url,
                               self.settings.max_pdf_bytes)
                return None
            try:
                title, sections = parse_pdf(response.content)
            except Exception as error:  # a corrupt PDF must not end the crawl
                logger.warning("Skipping %s: unreadable PDF (%s)", url, error)
                return None
            name = urlparse(url).path.rsplit("/", 1)[-1] or url
            document = Document(
                source=url, origin=self.source.name, kind="pdf", title=title or name,
                sections=sections, fetched_at=now,
            )
            return document, []
        logger.info("Skipping %s: content type %r", url, content_type)
        return None
