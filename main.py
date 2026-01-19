import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, List, Tuple
from urllib.parse import urlparse, urljoin, urldefrag, quote_plus

import dateparser
import httpx
from bs4 import BeautifulSoup


# ---------------------------
# Utilities
# ---------------------------

# Dateparser settings to standardize date extraction behavior.
DATE_SETTINGS = {
    "RETURN_AS_TIMEZONE_AWARE": False,
    "PREFER_DAY_OF_MONTH": "first",
    "PREFER_DATES_FROM": "past",
}

# Keywords to score "how to apply" links.
KEYWORDS_APPLY = [
    "apply", "application", "submit", "proposal", "request for proposals", "rfp",
    "funding opportunity", "opportunity", "how to apply", "apply now", "apply online",
    "online application", "application portal", "grant application", "portal",
    "online application portal", "grant portal"
]

# Text hints that indicate the page is missing.
NOT_FOUND_TEXT_HINTS = [
    "page not found", "404", "not found", "doesn't exist", "cannot be found"
]

# Match EINs in either 2-7 format or plain 9 digits.
EIN_REGEX = re.compile(r"\b(\d{2}-\d{7})\b|\b(\d{9})\b")


def canonicalize_url(u: str) -> str:
    # Remove fragment and normalize trailing slash (lightweight).
    u, _frag = urldefrag(u.strip())
    return u.rstrip("/")


def safe_iso_date(dt: Optional[datetime]) -> Optional[str]:
    # Convert a datetime to ISO date string, or keep None.
    if dt is None:
        return None
    return dt.date().isoformat()


def text_hash(s: str) -> str:
    # Short stable hash for text (useful for debugging/dedup).
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def normalize_whitespace(s: str) -> str:
    # Collapse repeated whitespace into a single space.
    return re.sub(r"\s+", " ", s).strip()


SUMMARY_KEYWORDS = [
    "mission", "purpose", "goal", "focus", "priority", "supports", "funding",
    "grant", "grants", "eligibility", "eligible", "application", "deadline",
    "community", "education", "health", "human services", "arts", "research",
]

EXCLUDE_SUMMARY_KEYWORDS = [
    "recent grants", "past grants", "grant recipients", "recipients",
    "awardees", "awards", "grants awarded", "recent awards",
]


def split_sentences(text: str) -> List[str]:
    # Split text into sentences using simple punctuation heuristics.
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def summarize_text(text: str, max_sentences: int = 4, max_chars: int = 900) -> str:
    # Summarize by selecting high-signal sentences (mission/eligibility/etc).
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return "NA"
    parts = split_sentences(cleaned)
    filtered = []
    for s in parts:
        lower = s.lower()
        if any(bad in lower for bad in EXCLUDE_SUMMARY_KEYWORDS):
            continue
        filtered.append(s)
    parts = filtered if filtered else parts
    if not parts:
        return cleaned[:max_chars]
    scored: List[Tuple[int, str]] = []
    for s in parts:
        lower = s.lower()
        score = 0
        for kw in SUMMARY_KEYWORDS:
            if kw in lower:
                score += 2
        if 40 <= len(s) <= 220:
            score += 1
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [s for _score, s in scored[:max_sentences] if _score > 0]
    if not selected:
        selected = parts[:max_sentences]
    summary = " ".join(selected)
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
    return summary


def looks_like_not_found(status_code: int, text: str) -> bool:
    # Detect a missing page by status code or textual hints.
    if status_code in (404, 410):
        return True
    lower = text.lower()
    return any(h in lower for h in NOT_FOUND_TEXT_HINTS)


def pick_best_title(soup: BeautifulSoup) -> str:
    # Priority: og:title -> h1 -> <title>.
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return normalize_whitespace(og["content"])

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return normalize_whitespace(h1.get_text(" ", strip=True))

    if soup.title and soup.title.get_text(strip=True):
        return normalize_whitespace(soup.title.get_text(" ", strip=True))

    return "NA"


def get_site_name(soup: BeautifulSoup, url: str) -> str:
    # Prefer og:site_name, then h1/title, fallback to domain.
    og_site = soup.find("meta", attrs={"property": "og:site_name"})
    if og_site and og_site.get("content"):
        return normalize_whitespace(og_site["content"])

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return normalize_whitespace(h1.get_text(" ", strip=True))

    if soup.title and soup.title.get_text(strip=True):
        return normalize_whitespace(soup.title.get_text(" ", strip=True))

    # fallback to domain
    host = urlparse(url).netloc
    host = host.replace("www.", "")
    return host if host else "NA"


def extract_main_text(soup: BeautifulSoup) -> str:
    # Remove common boilerplate tags.
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    # Prefer <main> or <article> if present.
    main = soup.find("main")
    if main:
        text = main.get_text(" ", strip=True)
        return normalize_whitespace(text)

    article = soup.find("article")
    if article:
        text = article.get_text(" ", strip=True)
        return normalize_whitespace(text)

    # Fallback: body text.
    body = soup.body
    if body:
        text = body.get_text(" ", strip=True)
        return normalize_whitespace(text)

    return normalize_whitespace(soup.get_text(" ", strip=True))


def find_best_application_link(soup: BeautifulSoup, base_url: str) -> str:
    # Score anchor tags to find the best "apply" link.
    candidates: List[Tuple[int, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        abs_url = urljoin(base_url, href)
        abs_url = canonicalize_url(abs_url)
        text_bits = [
            a.get_text(" ", strip=True),
            a.get("aria-label", ""),
            a.get("title", ""),
        ]
        anchor_text = normalize_whitespace(" ".join(t for t in text_bits if t)).lower()

        score = 0
        # Keyword hits in anchor text.
        for kw in KEYWORDS_APPLY:
            if kw in anchor_text:
                score += 3

        # Keyword hits in url path.
        lower_u = abs_url.lower()
        for kw in ["apply", "application", "submit", "rfp", "proposal"]:
            if kw in lower_u:
                score += 2

        # Deprioritize donation/contact/legal.
        for bad in ["donate", "contact", "privacy", "terms", "cookie", "unsubscribe"]:
            if bad in anchor_text or bad in lower_u:
                score -= 4

        # Prefer direct file downloads for applications.
        if re.search(r"\.(pdf|doc|docx)$", lower_u):
            score += 3

        # Button-like hints.
        class_attr = " ".join(a.get("class", []))
        if "btn" in class_attr.lower():
            score += 1

        if score > 0:
            candidates.append((score, abs_url))

    # Also consider <form action=...> as potential application portals.
    for form in soup.find_all("form"):
        action = (form.get("action") or "").strip()
        if not action:
            continue
        abs_url = canonicalize_url(urljoin(base_url, action))
        lower_u = abs_url.lower()
        score = 0
        for kw in ["apply", "application", "submit", "portal"]:
            if kw in lower_u:
                score += 3
        if score > 0:
            candidates.append((score, abs_url))

    if not candidates:
        return "NA"

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_navigation_candidates(soup: BeautifulSoup, base_url: str) -> List[str]:
    # Pick a few likely "more info" pages to follow when apply links are missing.
    keywords = [
        "learn more", "learn", "more", "details", "about",
        "funding", "grant", "grants", "guidelines", "opportunity",
    ]
    candidates: List[Tuple[int, str]] = []
    base_host = urlparse(base_url).netloc.replace("www.", "")

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        abs_url = canonicalize_url(urljoin(base_url, href))
        host = urlparse(abs_url).netloc.replace("www.", "")
        if host and host != base_host:
            continue

        anchor_text = normalize_whitespace(a.get_text(" ", strip=True)).lower()
        score = 0
        for kw in keywords:
            if kw in anchor_text:
                score += 2
        lower_u = abs_url.lower()
        for kw in ["apply", "application", "submit", "rfp", "proposal", "grant", "fund"]:
            if kw in lower_u:
                score += 1
        if score > 0:
            candidates.append((score, abs_url))

    candidates.sort(key=lambda x: x[0], reverse=True)
    # Return up to 3 unique URLs to limit extra fetches.
    uniq: List[str] = []
    seen = set()
    for _score, u in candidates:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
        if len(uniq) >= 3:
            break
    return uniq


def find_application_link_with_fallback(
    soup: BeautifulSoup,
    base_url: str,
    timeout: int,
) -> str:
    # First try to find an application link on the current page.
    direct = find_best_application_link(soup, base_url)
    if direct != "NA":
        return direct

    # If not found, follow a few likely navigation links and re-try.
    for nav_url in find_navigation_candidates(soup, base_url):
        status, final_url, html = fetch_html(nav_url, timeout=timeout)
        nav_soup = BeautifulSoup(html, "lxml")
        nav_text = extract_main_text(nav_soup)
        if looks_like_not_found(status, nav_text):
            continue
        candidate = find_best_application_link(nav_soup, final_url)
        if candidate != "NA":
            return candidate
    return "NA"


def find_ein_candidates(soup: BeautifulSoup, base_url: str) -> List[str]:
    # Pick a few in-domain pages likely to contain EIN/Tax ID details.
    keywords = ["ein", "tax id", "tax-id", "990", "contact", "about", "legal"]
    candidates: List[Tuple[int, str]] = []
    base_host = urlparse(base_url).netloc.replace("www.", "")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        abs_url = canonicalize_url(urljoin(base_url, href))
        host = urlparse(abs_url).netloc.replace("www.", "")
        if host and host != base_host:
            continue
        text = normalize_whitespace(a.get_text(" ", strip=True)).lower()
        score = 0
        for kw in keywords:
            if kw in text or kw in abs_url.lower():
                score += 2
        if score > 0:
            candidates.append((score, abs_url))
    candidates.sort(key=lambda x: x[0], reverse=True)
    uniq: List[str] = []
    seen = set()
    for _score, u in candidates:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
        if len(uniq) >= 3:
            break
    return uniq


def search_ein_external(org_name: str, domain: str, timeout: int) -> Optional[int]:
    # Fallback to external search if site doesn't publish EIN.
    query = org_name if org_name and org_name != "NA" else domain
    if not query:
        return None
    q = quote_plus(f"{query} EIN")
    url = f"https://duckduckgo.com/html/?q={q}"
    try:
        status, final_url, html = fetch_html(url, timeout=timeout)
    except Exception:
        return None
    if status != 200:
        return None
    soup = BeautifulSoup(html, "lxml")
    text = normalize_whitespace(soup.get_text(" ", strip=True))
    return extract_ein(text)


def find_ein_with_fallback(soup: BeautifulSoup, base_url: str, timeout: int) -> Optional[int]:
    # Try current page first, then a few likely in-domain pages.
    main_text = extract_main_text(soup)
    ein = extract_ein(main_text)
    if ein is not None:
        return ein
    for nav_url in find_ein_candidates(soup, base_url):
        status, final_url, html = fetch_html(nav_url, timeout=timeout)
        nav_soup = BeautifulSoup(html, "lxml")
        nav_text = extract_main_text(nav_soup)
        if looks_like_not_found(status, nav_text):
            continue
        ein = extract_ein(nav_text)
        if ein is not None:
            return ein
    domain = urlparse(base_url).netloc.replace("www.", "")
    return search_ein_external(get_site_name(soup, base_url), domain, timeout)


def build_description_summary(soup: BeautifulSoup, base_url: str, timeout: int) -> str:
    # Build a more comprehensive description by sampling a few related pages.
    texts = []
    main_text = extract_main_text(soup)
    if main_text:
        texts.append(main_text)

    for nav_url in find_navigation_candidates(soup, base_url):
        status, final_url, html = fetch_html(nav_url, timeout=timeout)
        nav_soup = BeautifulSoup(html, "lxml")
        nav_text = extract_main_text(nav_soup)
        if looks_like_not_found(status, nav_text):
            continue
        if nav_text:
            texts.append(nav_text)
        if len(texts) >= 3:
            break

    combined = " ".join(texts)
    return summarize_text(combined, max_sentences=4, max_chars=900)


def build_recovery_candidates(url: str) -> List[str]:
    # Build parent/root URLs for recovery within the same domain.
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    candidates: List[str] = []
    if parts:
        parent_path = "/".join(parts[:-1])
        parent_url = f"{parsed.scheme}://{parsed.netloc}"
        if parent_path:
            parent_url = f"{parent_url}/{parent_path}"
        candidates.append(parent_url)
    root_url = f"{parsed.scheme}://{parsed.netloc}"
    candidates.append(root_url)
    # Deduplicate while preserving order and remove original URL.
    seen = set()
    uniq = []
    for c in candidates:
        c = canonicalize_url(c)
        if c == canonicalize_url(url) or c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def find_similar_link_on_page(soup: BeautifulSoup, base_url: str, target_url: str) -> Optional[str]:
    # Try to find a link on the page that resembles the target URL.
    target_path = urlparse(target_url).path.rstrip("/")
    target_parts = [p for p in target_path.split("/") if p]
    slug = target_parts[-1] if target_parts else ""
    slug_words = [w for w in re.split(r"[-_]+", slug.lower()) if w]
    target_netloc = urlparse(target_url).netloc.replace("www.", "")

    best_score = 0
    best_url: Optional[str] = None
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        abs_url = canonicalize_url(urljoin(base_url, href))
        netloc = urlparse(abs_url).netloc.replace("www.", "")
        if netloc and netloc != target_netloc:
            continue
        anchor_text = normalize_whitespace(a.get_text(" ", strip=True)).lower()
        score = 0
        if slug and slug.lower() in abs_url.lower():
            score += 5
        if slug_words and all(w in anchor_text for w in slug_words):
            score += 3
        if score > best_score:
            best_score = score
            best_url = abs_url
    return best_url if best_score > 0 else None


def recover_from_not_found(url: str, timeout: int = 20) -> Optional[Tuple[int, str, str]]:
    # Try parent/root and in-domain similar links when a page is missing.
    for candidate in build_recovery_candidates(url):
        status, final_url, html = fetch_html(candidate, timeout=timeout)
        soup = BeautifulSoup(html, "lxml")
        main_text = extract_main_text(soup)
        if not looks_like_not_found(status, main_text):
            return status, canonicalize_url(final_url), html
        alt = find_similar_link_on_page(soup, candidate, url)
        if alt:
            status2, final_url2, html2 = fetch_html(alt, timeout=timeout)
            soup2 = BeautifulSoup(html2, "lxml")
            main_text2 = extract_main_text(soup2)
            if not looks_like_not_found(status2, main_text2):
                return status2, canonicalize_url(final_url2), html2
    return None


def extract_ein(text: str) -> Optional[int]:
    """
    Return EIN as int if confidently found, else None.
    Accepts either XX-XXXXXXX or 9 digits.
    """
    matches = EIN_REGEX.findall(text)
    if not matches:
        return None

    # matches is list of tuples (with possible groups).
    for g1, g2 in matches:
        raw = g1 or g2
        if not raw:
            continue
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 9:
            # Avoid phone numbers by simple heuristic; return first 9 digits.
            return int(digits)
    return None


def parse_date_from_context(full_text: str, labels: List[str]) -> Optional[datetime]:
    """
    Try to find a date near any of the given labels. If not found, return None.
    """
    lower = full_text.lower()
    sentences = split_sentences(full_text)
    for label in labels:
        if label not in lower:
            continue
        # First try sentence-level parsing.
        for s in sentences:
            if label in s.lower():
                dt = dateparser.parse(s, settings=DATE_SETTINGS)
                if dt:
                    return dt
                # Fallback: month/day without year (e.g., "March 15, each year").
                m = re.search(
                    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
                    r"dec(?:ember)?)\s+(\d{1,2})(?:\s*,\s*(\d{4}))?",
                    s,
                    flags=re.IGNORECASE,
                )
                if m:
                    month_str, day_str, year_str = m.groups()
                    year = int(year_str) if year_str else datetime.now().year
                    dt = dateparser.parse(
                        f"{month_str} {day_str} {year}",
                        settings=DATE_SETTINGS,
                    )
                    if dt:
                        return dt
        # Fallback: parse date in a small context window around the label.
        idx = lower.find(label)
        window = full_text[max(0, idx - 80): idx + 200]
        dt = dateparser.parse(window, settings=DATE_SETTINGS)
        if dt:
            return dt
    return None


def parse_any_date(full_text: str) -> Optional[datetime]:
    """
    Last-resort date parsing: try parsing from the first few hundred chars.
    """
    snippet = full_text[:400]
    return dateparser.parse(snippet, settings=DATE_SETTINGS)


def guess_notice_id(soup: BeautifulSoup, text: str, ein: Optional[int], index: int, url: str) -> str:
    # Try to find an explicit ID-like pattern in text.
    patterns = [
        r"(Notice\s*(ID|No\.?)\s*[:#]?\s*([A-Za-z0-9\-_./]+))",
        r"(Opportunity\s*(Number|No\.?)\s*[:#]?\s*([A-Za-z0-9\-_./]+))",
        r"(FOA\s*[:#]?\s*([A-Za-z0-9\-_./]+))",
        r"(RFP\s*[:#]?\s*([A-Za-z0-9\-_./]+))",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            # last capturing group tends to be the id
            candidate = m.groups()[-1]
            candidate = normalize_whitespace(candidate)
            if candidate and len(candidate) <= 64:
                return candidate

    # Fallback per spec: [EIN]_[Index].
    # If EIN missing, use 0 to keep deterministic and schema-friendly.
    ein_part = str(ein) if ein is not None else "0"
    return f"{ein_part}_{index}"


# ---------------------------
# Data model
# ---------------------------

@dataclass
class GrantRecord:
    # Output schema for each grant record.
    agency: str
    application_link: str
    description: str
    ein: Optional[int]
    last_update: Optional[str]
    name: str
    notice_id: str
    organizations: str
    published_date: Optional[str]
    response_date: Optional[str]
    url: str


# ---------------------------
# Core extraction
# ---------------------------

def fetch_html(url: str, timeout: int = 20) -> Tuple[int, str, str]:
    """
    Return (status_code, final_url, html_text).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GrantScraper/1.0; +https://example.com/bot)"
    }
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        # Use follow_redirects to normalize to the final URL.
        resp = client.get(url)
        return resp.status_code, str(resp.url), resp.text


def extract_one(url: str, index: int = 1, timeout: int = 20) -> Tuple[GrantRecord, dict]:
    # Extract one grant record from a single URL with meta info.
    orig_url = canonicalize_url(url)
    status, final_url, html = fetch_html(orig_url, timeout=timeout)
    final_url = canonicalize_url(final_url)

    soup = BeautifulSoup(html, "lxml")
    title = pick_best_title(soup)
    site_name = get_site_name(soup, final_url)
    main_text = extract_main_text(soup)

    # Detect not found.
    if looks_like_not_found(status, main_text):
        # Try recovery within the same domain before giving up.
        recovered = recover_from_not_found(final_url, timeout=timeout)
        if recovered:
            status, final_url, html = recovered
            soup = BeautifulSoup(html, "lxml")
            title = pick_best_title(soup)
            site_name = get_site_name(soup, final_url)
            main_text = extract_main_text(soup)
        else:
            # Still output a valid schema line (robustness).
            ein = None
            notice_id = guess_notice_id(soup, main_text, ein, index, final_url)
            record = GrantRecord(
                agency=site_name,
                application_link="NA",
                description="Page appears to be missing (404 / Not Found) or content could not be extracted.",
                ein=None,
                last_update=None,
                name=title if title != "NA" else "NA",
                notice_id=notice_id,
                organizations=site_name,
                published_date=None,
                response_date=None,
                url=final_url,
            )
            return record, {
                "input_url": orig_url,
                "final_url": final_url,
                "status": "fail",
                "reason": "not_found",
            }

    ein = find_ein_with_fallback(soup, final_url, timeout)

    # Dates (heuristic labels).
    published_dt = parse_date_from_context(main_text, ["posted", "published", "date:", "posted on", "publication date"])
    updated_dt = parse_date_from_context(main_text, ["last updated", "updated", "modified"])
    deadline_dt = parse_date_from_context(
        main_text,
        ["deadline", "due", "applications due", "response date", "closing date", "application deadline", "deadline for applications"]
    )

    # If nothing found, keep nulls (don’t guess wildly).
    published_date = safe_iso_date(published_dt)
    last_update = safe_iso_date(updated_dt)
    response_date = safe_iso_date(deadline_dt)

    application_link = find_application_link_with_fallback(soup, final_url, timeout)

    # Description: summarize across a few related pages for better coverage.
    desc = build_description_summary(soup, final_url, timeout)

    # Organizations + agency: MVP uses site_name; later can improve with page cues.
    organizations = site_name
    agency = site_name

    notice_id = guess_notice_id(soup, main_text, ein, index, final_url)

    # Name: use title; fallback to site_name + " Grant Opportunity".
    name = title if title != "NA" else f"{site_name} Grant Opportunity"

    record = GrantRecord(
        agency=agency,
        application_link=application_link,
        description=desc if desc else "NA",
        ein=ein,
        last_update=last_update,
        name=name,
        notice_id=notice_id,
        organizations=organizations,
        published_date=published_date,
        response_date=response_date,
        url=final_url,
    )
    return record, {
        "input_url": orig_url,
        "final_url": final_url,
        "status": "success",
        "reason": "ok",
    }


# ---------------------------
# CLI
# ---------------------------

def read_urls_from_file(path: str) -> List[str]:
    # Extract URLs from a text file using regex.
    urls = []
    url_pattern = re.compile(r"https?://[^\s\)\"\']+")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            for match in url_pattern.findall(line):
                urls.append(match.strip())
    return urls


def write_jsonl(records: List[GrantRecord], out_path: str) -> None:
    # Write records to JSONL file.
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def get_missing_fields(record: GrantRecord) -> List[str]:
    # Identify which fields are missing (None or "NA").
    missing = []
    data = asdict(record)
    for k, v in data.items():
        if v is None:
            missing.append(k)
        elif isinstance(v, str) and v.strip().upper() == "NA":
            missing.append(k)
    return missing


def write_run_report(
    output_path: str,
    per_url: List[dict],
    missing_field_counts: dict,
    not_found_urls: List[str],
    duplicate_urls: List[str],
) -> str:
    # Write a run report file next to the output JSONL.
    output_name = os.path.basename(output_path)
    report_name = f"report for output {output_name}.json"
    report_path = os.path.join(os.path.dirname(output_path), report_name)
    report = {
        "title": f"report for output {output_name}",
        "output_file": output_name,
        "total": len(per_url),
        "success_count": sum(1 for r in per_url if r["status"] == "success"),
        "fail_count": sum(1 for r in per_url if r["status"] == "fail"),
        "missing_field_counts": missing_field_counts,
        "not_found_urls": not_found_urls,
        "duplicate_urls": duplicate_urls,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def main():
    # CLI entrypoint and batch driver.
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, help="Single URL to process")
    ap.add_argument("--input", type=str, help="Path to a txt file containing URLs (one per line)")
    ap.add_argument("--output", type=str, default="output.jsonl", help="Output JSONL file path")
    ap.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    args = ap.parse_args()

    # If user didn't override output, add a timestamp to avoid overwriting.
    if args.output == "output.jsonl":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output_{ts}.jsonl"

    if not args.url and not args.input:
        print("Provide --url or --input", file=sys.stderr)
        sys.exit(2)

    urls: List[str] = []
    if args.url:
        urls = [args.url]
    else:
        urls = read_urls_from_file(args.input)

    # Deduplicate URLs while preserving first occurrence.
    seen = set()
    unique_urls: List[str] = []
    duplicate_urls: List[str] = []
    duplicate_seen = set()
    for u in urls:
        cu = canonicalize_url(u)
        if cu in seen:
            if cu not in duplicate_seen:
                duplicate_urls.append(u)
                duplicate_seen.add(cu)
            continue
        seen.add(cu)
        unique_urls.append(u)
    urls = unique_urls

    records: List[GrantRecord] = []
    per_url: List[dict] = []
    missing_field_counts = {k: 0 for k in [
        "agency", "application_link", "description", "ein", "last_update", "name",
        "notice_id", "organizations", "published_date", "response_date", "url",
    ]}
    not_found_urls: List[str] = []

    for i, u in enumerate(urls, start=1):
        try:
            start = time.perf_counter()
            rec, meta = extract_one(u, index=i, timeout=args.timeout)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            records.append(rec)
            missing = get_missing_fields(rec)
            for k in missing:
                if k in missing_field_counts:
                    missing_field_counts[k] += 1
            if meta.get("reason") == "not_found":
                not_found_urls.append(meta.get("input_url", u))
            per_url.append({
                "input_url": meta.get("input_url", canonicalize_url(u)),
                "final_url": meta.get("final_url", rec.url),
                "status": meta.get("status", "success"),
                "reason": meta.get("reason", "ok"),
                "extracted_fields_missing": missing,
                "time_ms": elapsed_ms,
            })
            print(f"[OK] {i}/{len(urls)} {u}")
        except Exception as e:
            # Robustness: don’t crash the batch.
            print(f"[FAIL] {i}/{len(urls)} {u} -> {e}", file=sys.stderr)
            # Still output a placeholder record.
            safe_u = canonicalize_url(u)
            rec = GrantRecord(
                agency="NA",
                application_link="NA",
                description=f"Failed to fetch or parse: {type(e).__name__}",
                ein=None,
                last_update=None,
                name="NA",
                notice_id=f"0_{i}",
                organizations="NA",
                published_date=None,
                response_date=None,
                url=safe_u,
            )
            records.append(rec)
            missing = get_missing_fields(rec)
            for k in missing:
                if k in missing_field_counts:
                    missing_field_counts[k] += 1
            reason = "timeout" if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout)) else "parse_error"
            per_url.append({
                "input_url": safe_u,
                "final_url": safe_u,
                "status": "fail",
                "reason": reason,
                "extracted_fields_missing": missing,
                "time_ms": None,
            })

    # Persist output for all URLs.
    write_jsonl(records, args.output)
    report_path = write_run_report(args.output, per_url, missing_field_counts, not_found_urls, duplicate_urls)
    print(f"Written: {args.output} (lines={len(records)})")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()