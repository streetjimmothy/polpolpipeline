import re
import sys
import time
import asyncio
import random
from urllib.parse import urlparse
import httpx 
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

#find_module shim. (used by snscrape, deprecated in python3.13)
import importlib.machinery as m
if not hasattr(m.FileFinder, "find_module"):
    def _find_module(self, fullname, path=None):
        spec = self.find_spec(fullname)
        return spec.loader if spec else None
    m.FileFinder.find_module = _find_module

import snscrape.modules.twitter as sntwitter


# --- Config tunables ---
MAX_CONCURRENCY = 256            # Max simultaneous in-flight URL resolutions
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 10
MAX_REDIRECTS = 10
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES_PER_URL = 10
TWEET_SCRAPE_TIMEOUT = 20
RESOLVE_HEAD_FIRST = True
BACKOFF_BASE = 0.25              # seconds (initial backoff)
BACKOFF_MAX = 8.0                # cap on backoff
JITTER_FRAC = 0.25               # random jitter fraction

# --- Caches ---
_URL_CACHE: dict[str, str] = {}
_URL_FAILS: dict[str, int] = {}
_TWEET_CACHE: dict[str, dict] = {}

# Precompiled regex (t.co + legacy X permalink forms)
URL_PATTERN = re.compile(r'https?://t\.co/\w+|https?://x\.com/i/web/status/\d+')

#############################
# Lazy async HTTP client (httpx.AsyncClient)
#############################
_async_client: httpx.AsyncClient | None = None

def _get_async_client() -> httpx.AsyncClient:
	global _async_client
	if _async_client is None:
		_async_client = httpx.AsyncClient(
			headers={
				"User-Agent": (
					"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
					"(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.3"
				)
			},
			timeout=httpx.Timeout(
				connect=CONNECT_TIMEOUT,
				read=READ_TIMEOUT,
				write=READ_TIMEOUT,
				pool=CONNECT_TIMEOUT,
			),
			limits=httpx.Limits(
				max_connections=MAX_CONCURRENCY,
				max_keepalive_connections=MAX_CONCURRENCY,
				keepalive_expiry=30.0,
			),
			follow_redirects=True,
			http2=True,
		)
	return _async_client


async def _backoff_delay(attempt: int) -> float:
	base = BACKOFF_BASE * (2 ** (attempt - 1))
	delay = min(base, BACKOFF_MAX)
	if JITTER_FRAC:
		delay += delay * JITTER_FRAC * random.random()
	return delay


# --- Resolve shortened / redirecting URLs (t.co, etc.) ---
async def resolve_url(url: str) -> str:
	if url in _URL_CACHE:
		return _URL_CACHE[url]
	if _URL_FAILS.get(url, 0) >= MAX_RETRIES_PER_URL:
		return url

	client = _get_async_client()
	attempt = 0
	while attempt < MAX_RETRIES_PER_URL:
		attempt += 1
		try:
			if RESOLVE_HEAD_FIRST:
				resp = await client.head(url)
				if resp.status_code in (405, 403) or resp.status_code >= 400:
					resp = await client.get(url)
			else:
				resp = await client.get(url)

			if resp.status_code in RETRY_STATUSES:
				_URL_FAILS[url] = _URL_FAILS.get(url, 0) + 1
				await asyncio.sleep(await _backoff_delay(attempt))
				continue

			final_url = str(resp.url)
			_URL_CACHE[url] = final_url
			return final_url
		except httpx.HTTPError:
			_URL_FAILS[url] = _URL_FAILS.get(url, 0) + 1
			await asyncio.sleep(await _backoff_delay(attempt))
			continue
		except Exception:
			_URL_FAILS[url] = _URL_FAILS.get(url, 0) + 1
			await asyncio.sleep(await _backoff_delay(attempt))
			continue

	_URL_CACHE[url] = url
	return url

# --- Extract domain from URL ---
def extract_host_domain(full_url):
	try:
		parsed = urlparse(full_url)
		return parsed.netloc.lower().removeprefix('www.')
	except Exception:
		return ""


def _scrape_tweet_sync(tweet_id: str):
	if tweet_id in _TWEET_CACHE:
		return _TWEET_CACHE[tweet_id]
	start = time.time()
	try:
		scraper = sntwitter.TwitterTweetScraper(tweet_id)
		for tweet in scraper.get_items():
			info = {"handle": tweet.user.username.lower(), "text": tweet.content}
			_TWEET_CACHE[tweet_id] = info
			return info
	except Exception:
		pass
	finally:
		if time.time() - start > TWEET_SCRAPE_TIMEOUT:
			return None
	_TWEET_CACHE[tweet_id] = None
	return None


async def get_tweet_info_from_id(tweet_id: str):
	if tweet_id in _TWEET_CACHE:
		return _TWEET_CACHE[tweet_id]
	return await asyncio.to_thread(_scrape_tweet_sync, tweet_id)

def _collect_unique_urls(infile_path: str) -> tuple[list[str], list[str]]:
	tco = []
	tweet_urls = []
	seen = set()
	with open(infile_path, 'r', encoding='utf-8', errors='ignore') as fh:
		for line in fh:
			for url in URL_PATTERN.findall(line):
				if url in seen:
					continue
				seen.add(url)
				if 'x.com/i/web/status/' in url:
					tweet_urls.append(url)
				else:
					tco.append(url)
	return tco, tweet_urls

async def _resolve_all_async(urls: list[str]) -> None:
	if not urls:
		return
	sem = asyncio.Semaphore(MAX_CONCURRENCY)

	async def worker(u: str):
		async with sem:
			await resolve_url(u)

	# Use tqdm over tasks; tqdm_asyncio.gather provides progress tracking
	await tqdm_asyncio.gather(*(worker(u) for u in urls), total=len(urls), desc="Resolving URLs", ncols=100)


async def _resolve_tweets_async(tweet_placeholder_urls: list[str]) -> None:
	if not tweet_placeholder_urls:
		return
	sem = asyncio.Semaphore(16)

	async def worker(u: str):
		tweet_id = u.rsplit('/', 1)[-1]
		async with sem:
			await get_tweet_info_from_id(tweet_id)

	await tqdm_asyncio.gather(*(worker(u) for u in tweet_placeholder_urls), total=len(tweet_placeholder_urls), desc="Scraping Tweets", ncols=100)


def _rewrite_line(line: str) -> str:
    urls = URL_PATTERN.findall(line)
    if not urls:
        return line
    for url in urls:
        if 'x.com/i/web/status/' in url:
            tweet_id = url.rsplit('/', 1)[-1]
            info = _TWEET_CACHE.get(tweet_id)
            if info:
                line = line.replace(url, f"https://x.com/{info['handle']}/status/{tweet_id}")
            else:
                line = line.replace(url, f"https://x.com/i/web/status/{tweet_id} [UNAVAILABLE]")
        else:
            # Already resolved + cached during async phase
            resolved = _URL_CACHE.get(url, url)
            line = line.replace(url, resolved)
    return line


async def process_file_async(infile: str, outfile: str):
	# Phase 1: gather unique URLs
	tco_urls, tweet_urls = _collect_unique_urls(infile)

	# Phase 2: async resolve short URLs
	await _resolve_all_async(tco_urls)

	# Phase 3: async tweet enrichment (scrape offloaded to threads)
	await _resolve_tweets_async(tweet_urls)

	# Phase 4: rewrite output
	with open(infile, 'r', encoding='utf-8', errors='ignore') as fin:
		lines = fin.readlines()

	with open(outfile, 'w', encoding='utf-8', newline='') as fout:
		for line in tqdm(lines, desc="Rewriting", ncols=100):
			fout.write(_rewrite_line(line))

	# Graceful close of async client
	if _async_client is not None:
		await _async_client.aclose()


def process_file(infile: str, outfile: str):
	"""Synchronous facade for CLI usage."""
	asyncio.run(process_file_async(infile, outfile))


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: python resolve_URLs.py <input> <output>", file=sys.stderr)
		sys.exit(1)
	process_file(sys.argv[1], sys.argv[2])