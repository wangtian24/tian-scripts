import asyncio
import logging
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag

FETCH_HTML_TIMEOUT = 10.0

# TODO: Make this more dynamic or use a webscraping paid solution
# Ref - https://scrapeops.io/web-scraping-playbook/403-forbidden-error-web-scraping/
browser_like_headers = {
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
    "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "macOS",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/99.0.4844.83 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.9"
    ),
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
}


async def get_html_from_url(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(
            timeout=FETCH_HTML_TIMEOUT,
            headers=browser_like_headers,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.TimeoutException:
        logging.warning(f"Citations: Timeout while fetching URL: {url}")
    except httpx.HTTPStatusError as e:
        logging.warning(f"Citations: HTTP error {e.response.status_code} while fetching URL: {url}")
    except Exception as e:
        logging.warning(f"Citations: Could not fetch URL {url}: {str(e)}")
    return None


async def get_html_from_urls(urls: list[str]) -> list[str]:
    try:
        tasks = [get_html_from_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return ["" if not isinstance(r, str) else r for r in results]
    except Exception as e:
        logging.warning(f"Citations: Could not fetch multiple URLs: {str(e)}")
        return []


async def get_title_and_description_from_html(html: str) -> tuple[str, str]:
    try:
        if not html:
            return "", ""

        soup = BeautifulSoup(html, "html.parser")

        title = None
        for title_tag in ["twitter:title", "og:title"]:
            meta_title = soup.find("meta", attrs={"property": title_tag}) or soup.find(
                "meta", attrs={"name": title_tag}
            )
            if isinstance(meta_title, Tag):
                content = meta_title.get("content")
                if isinstance(content, str):
                    title = content.strip()
                    break

        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()

        description = None
        for desc_tag in ["twitter:description", "og:description", "description"]:
            meta_desc = soup.find("meta", attrs={"property": desc_tag}) or soup.find("meta", attrs={"name": desc_tag})
            if isinstance(meta_desc, Tag):
                content = meta_desc.get("content")
                if isinstance(content, str):
                    description = content.strip()
                    break

        return title or "", description or ""
    except Exception as e:
        logging.warning(f"Citations: Could not parse HTML: {str(e)}")
        return "", ""


async def enhance_citations(citations: list[str]) -> list[dict[str, str]]:
    try:
        start_time = datetime.now()
        if not citations:
            return []

        htmls = await get_html_from_urls(citations)
        if not htmls:
            return [{"title": get_domain_from_url(url), "description": "", "url": url} for url in citations]

        title_and_descriptions = await asyncio.gather(
            *[get_title_and_description_from_html(html) for html in htmls], return_exceptions=True
        )

        enhanced_citations = []
        for result, url in zip(title_and_descriptions, citations, strict=True):
            if isinstance(result, BaseException):
                logging.warning(f"Citations: Could not process {url}: {str(result)}")
                enhanced_citations.append({"title": get_domain_from_url(url), "description": "", "url": url})
            else:
                title, description = result
                enhanced_citations.append({"title": title, "description": description, "url": url})

        end_time = datetime.now()
        logging.info(f"Citations: Time taken to parse citations: {end_time - start_time}")
        return enhanced_citations
    except Exception as e:
        logging.warning(f"Citations: Could not enhance citations: {str(e)}")
        return [{"title": get_domain_from_url(url), "description": "", "url": url} for url in citations]


def get_domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception as e:
        logging.warning(f"Citations: Could not get domain from URL: {url} - {str(e)}")
        return url
