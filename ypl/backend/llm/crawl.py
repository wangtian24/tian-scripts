import asyncio
import logging
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

FETCH_HTML_TIMEOUT = 10.0


async def get_html_from_url(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=FETCH_HTML_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()  # Raises an exception for 4XX/5XX status codes
            return response.text
    except httpx.TimeoutException:
        logging.error(f"Timeout while fetching URL: {url}")
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error {e.response.status_code} while fetching URL: {url}")
    except Exception as e:
        logging.error(f"Error fetching URL {url}: {str(e)}")
    return None


async def get_html_from_urls(urls: list[str]) -> list[str]:
    try:
        tasks = [get_html_from_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return ["" if not isinstance(r, str) else r for r in results]
    except Exception as e:
        logging.error(f"Error fetching multiple URLs: {str(e)}")
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
            if meta_title and meta_title.get("content"):
                title = meta_title["content"].strip()
                break

        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()

        description = None
        for desc_tag in ["twitter:description", "og:description", "description"]:
            meta_desc = soup.find("meta", attrs={"property": desc_tag}) or soup.find("meta", attrs={"name": desc_tag})
            if meta_desc and meta_desc.get("content"):
                description = meta_desc["content"].strip()
                break

        return title or "", description or ""
    except Exception as e:
        logging.error(f"Error parsing HTML: {str(e)}")
        return "", ""


async def enhance_citations(citations: list[str]) -> list[dict[str, str]]:
    try:
        start_time = datetime.now()
        if not citations:
            return []

        htmls = await get_html_from_urls(citations)
        if not htmls:
            return [{"title": "", "description": "", "url": url} for url in citations]

        title_and_descriptions = await asyncio.gather(
            *[get_title_and_description_from_html(html) for html in htmls], return_exceptions=True
        )

        enhanced_citations = []
        for result, url in zip(title_and_descriptions, citations, strict=True):
            if isinstance(result, BaseException):
                logging.error(f"Error processing {url}: {str(result)}")
                enhanced_citations.append({"title": "", "description": "", "url": url})
            else:
                title, description = result
                enhanced_citations.append({"title": title, "description": description, "url": url})

        end_time = datetime.now()
        logging.info(f"Time taken to parse citations: {end_time - start_time}")
        return enhanced_citations
    except Exception as e:
        logging.error(f"Error enhancing citations: {str(e)}")
        return [{"title": "", "description": "", "url": url} for url in citations]
