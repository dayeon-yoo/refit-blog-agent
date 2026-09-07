from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from config.settings import get_settings
from agents.tag_agent import KeywordDataProvider


class NaverKeywordProviderError(RuntimeError):
    """Raised when the Naver Search Ads provider cannot return metrics."""

    def __init__(self, message: str, partial_metrics: Optional[Dict[str, Dict]] = None):
        super().__init__(message)
        self.partial_metrics = partial_metrics


class NaverKeywordProvider(KeywordDataProvider):
    BASE_URL = "https://api.searchad.naver.com"
    URI = "/keywordstool"
    METHOD = "GET"
    # keywordstool may return related terms rather than every hint when a
    # large comma-separated batch is submitted. Keep batches small so each
    # requested candidate has a reliable chance to be matched.
    MAX_HINT_KEYWORDS = 5
    COMPETITION_MAP = {
        "low": 0.2,
        "낮음": 0.2,
        "medium": 0.5,
        "중간": 0.5,
        "high": 0.9,
        "높음": 0.9,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        customer_id: Optional[str] = None,
        *,
        timeout: float = 10.0,
        max_retries: int = 1,
        clock_ms: Optional[Callable[[], int]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        requester: Optional[Callable[[str, Dict[str, str], float], Tuple]] = None,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.naver_searchad_api_key
        self.secret_key = secret_key if secret_key is not None else settings.naver_searchad_secret_key
        self.customer_id = customer_id if customer_id is not None else settings.naver_searchad_customer_id
        if not all((self.api_key, self.secret_key, self.customer_id)):
            raise ValueError("Naver Search Ads API credentials are not configured.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.timeout = timeout
        self.max_retries = max_retries
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.sleep_fn = sleep_fn
        self.requester = requester or self._request

    @staticmethod
    def generate_signature(timestamp: str, method: str, uri: str, secret_key: str) -> str:
        """Generate the official Search Ads API Base64 HMAC-SHA256 signature."""
        message = f"{timestamp}.{method}.{uri}"
        digest = hmac.new(
            secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("ascii")

    def _headers(self, timestamp: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json; charset=UTF-8",
            "X-Timestamp": timestamp,
            "X-API-KEY": self.api_key,
            "X-Customer": str(self.customer_id),
            "X-Signature": self.generate_signature(timestamp, self.METHOD, self.URI, self.secret_key),
        }

    def _request(self, url: str, headers: Dict[str, str], timeout: float) -> Tuple:
        request = urllib.request.Request(url, headers=headers, method=self.METHOD)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return int(response.getcode()), response.read(), dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read(), dict(exc.headers.items()) if exc.headers else {}

    @staticmethod
    def _normalize_keyword(value: object) -> str:
        value = unicodedata.normalize("NFKC", str(value or "")).casefold()
        return "".join(ch for ch in value if ch.isalnum())

    @staticmethod
    def _parse_count(value: object) -> int:
        if isinstance(value, (int, float)):
            return max(0, int(value))
        text = str(value or "").strip().replace(",", "")
        if not text or text in {"-", "N/A", "null"}:
            return 0
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else 0

    @classmethod
    def _competition(cls, value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        text = str(value).strip().lower()
        if text in cls.COMPETITION_MAP:
            return cls.COMPETITION_MAP[text]
        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            return None

    def _build_url(self, keywords: List[str]) -> str:
        query = urllib.parse.urlencode({
            "hintKeywords": ",".join(keywords),
            "showDetail": "1",
        })
        return f"{self.BASE_URL}{self.URI}?{query}"

    @staticmethod
    def _retry_delay(response_headers: Dict[str, str], attempt: int) -> float:
        retry_after = response_headers.get("Retry-After") or response_headers.get("retry-after")
        try:
            if retry_after is not None:
                return min(60.0, max(1.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
        return min(60.0, 5.0 * (2**attempt))

    def _fetch_payload(self, keywords: List[str]) -> Dict:
        url = self._build_url(keywords)
        for attempt in range(self.max_retries + 1):
            timestamp = str(self.clock_ms())
            try:
                response = self.requester(url, self._headers(timestamp), self.timeout)
                status, body = response[0], response[1]
                response_headers = response[2] if len(response) > 2 else {}
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                if attempt < self.max_retries:
                    self.sleep_fn(self._retry_delay({}, attempt))
                    continue
                raise NaverKeywordProviderError("Naver Search Ads API request timed out or failed.") from exc

            if status == 200:
                try:
                    payload = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
                except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise NaverKeywordProviderError("Naver Search Ads API returned malformed JSON.") from exc
                if not isinstance(payload, dict) or not isinstance(payload.get("keywordList"), list):
                    raise NaverKeywordProviderError("Naver Search Ads API response has no keywordList.")
                return payload

            if status in {429} or 500 <= status <= 599:
                if attempt < self.max_retries:
                    self.sleep_fn(self._retry_delay(response_headers, attempt))
                    continue
                raise NaverKeywordProviderError(f"Naver Search Ads API temporary error: HTTP {status}")

            if status in {401, 403}:
                raise NaverKeywordProviderError(f"Naver Search Ads API authentication failed: HTTP {status}")
            raise NaverKeywordProviderError(f"Naver Search Ads API request failed: HTTP {status}")

        raise NaverKeywordProviderError("Naver Search Ads API request failed.")

    def fetch_metrics(self, keywords: List[str]) -> Dict[str, Dict]:
        requested: List[str] = []
        seen = set()
        for keyword in keywords:
            cleaned = " ".join(str(keyword or "").split()).strip()
            normalized = self._normalize_keyword(cleaned)
            if cleaned and normalized not in seen:
                requested.append(cleaned)
                seen.add(normalized)
        if not requested:
            return {}

        requested_by_normalized = {
            self._normalize_keyword(keyword): keyword for keyword in requested
        }
        metrics = {
            keyword: {
                "search_volume": None,
                "competition": None,
                "publishing_volume": None,
                "saturation": None,
                "metrics_available": False,
            }
            for keyword in requested
        }

        for start in range(0, len(requested), self.MAX_HINT_KEYWORDS):
            try:
                payload = self._fetch_payload(requested[start : start + self.MAX_HINT_KEYWORDS])
            except NaverKeywordProviderError as exc:
                raise NaverKeywordProviderError(str(exc), partial_metrics=metrics) from exc
            for item in payload["keywordList"]:
                if not isinstance(item, dict):
                    continue
                response_keyword = self._normalize_keyword(item.get("relKeyword"))
                requested_keyword = requested_by_normalized.get(response_keyword)
                if not requested_keyword:
                    continue
                metrics[requested_keyword] = {
                    "search_volume": self._parse_count(item.get("monthlyPcQcCnt"))
                    + self._parse_count(item.get("monthlyMobileQcCnt")),
                    "competition": self._competition(item.get("compIdx")),
                    "publishing_volume": None,
                    "saturation": None,
                    "metrics_available": True,
                }
        return metrics
