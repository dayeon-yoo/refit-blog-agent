from __future__ import annotations

import base64
import hashlib
import hmac
import json
import urllib.parse

import pytest

from agents.tag_agent import MockKeywordDataProvider
from config.settings import get_settings
from orchestrator.workflow import build_keyword_provider
from providers.naver_keyword_provider import NaverKeywordProvider, NaverKeywordProviderError


def make_provider(requester, **kwargs):
    return NaverKeywordProvider(
        api_key="api-key",
        secret_key="secret-key",
        customer_id="12345",
        requester=requester,
        clock_ms=lambda: 1700000000000,
        sleep_fn=lambda _seconds: None,
        **kwargs,
    )


def test_signature_matches_official_hmac_sha256_flow():
    expected = base64.b64encode(
        hmac.new(
            b"secret-key",
            b"1700000000000.GET./keywordstool",
            hashlib.sha256,
        ).digest()
    ).decode("ascii")

    assert NaverKeywordProvider.generate_signature(
        "1700000000000", "GET", "/keywordstool", "secret-key"
    ) == expected


def test_fetch_metrics_maps_response_and_builds_query_and_headers():
    captured = {}

    def requester(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return 200, json.dumps({
            "keywordList": [
                {
                    "relKeyword": "가을 옷장 정리",
                    "monthlyPcQcCnt": "< 10",
                    "monthlyMobileQcCnt": "1,200",
                    "compIdx": "HIGH",
                },
                {
                    "relKeyword": "옷장 수납",
                    "monthlyPcQcCnt": 300,
                    "monthlyMobileQcCnt": 400,
                    "compIdx": "낮음",
                },
            ]
        }).encode("utf-8")

    provider = make_provider(requester)
    result = provider.fetch_metrics(["가을 옷장 정리", "옷장 수납"])
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(captured["url"]).query)

    assert query["hintKeywords"] == ["가을 옷장 정리,옷장 수납"]
    assert query["showDetail"] == ["1"]
    assert captured["headers"]["X-Timestamp"] == "1700000000000"
    assert captured["headers"]["X-API-KEY"] == "api-key"
    assert captured["headers"]["X-Customer"] == "12345"
    assert captured["headers"]["X-Signature"] == NaverKeywordProvider.generate_signature(
        "1700000000000", "GET", "/keywordstool", "secret-key"
    )
    assert captured["timeout"] == 10.0
    assert result["가을 옷장 정리"]["search_volume"] == 1210
    assert result["가을 옷장 정리"]["competition"] == 0.9
    assert result["옷장 수납"]["competition"] == 0.2
    assert result["가을 옷장 정리"]["publishing_volume"] is None
    assert result["가을 옷장 정리"]["saturation"] is None


def test_missing_requested_keyword_is_returned_with_neutral_metrics():
    def requester(_url, _headers, _timeout):
        return 200, b'{"keywordList": [{"relKeyword": "other", "monthlyPcQcCnt": 10, "monthlyMobileQcCnt": 20, "compIdx": "LOW"}]}'

    result = make_provider(requester).fetch_metrics(["missing"])

    assert result == {
        "missing": {
            "search_volume": None,
            "competition": None,
            "publishing_volume": None,
            "saturation": None,
            "metrics_available": False,
        }
    }


def test_large_keyword_batches_are_merged_without_losing_requested_metrics():
    calls = []

    def requester(url, _headers, _timeout):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        hints = query["hintKeywords"][0].split(",")
        calls.append(hints)
        return 200, json.dumps({
            "keywordList": [{
                    "relKeyword": f"{hint[:2]} {hint[2:]}",
                    "monthlyPcQcCnt": 10,
                    "monthlyMobileQcCnt": 20,
                    "compIdx": "HIGH",
                } for hint in hints
            ]
        }).encode("utf-8")

    keywords = [f"keyword-{index}" for index in range(12)]
    result = make_provider(requester).fetch_metrics(keywords)

    assert [len(batch) for batch in calls] == [5, 5, 2]
    assert result["keyword-0"]["search_volume"] == 30
    assert result["keyword-0"]["competition"] == 0.9
    assert result["keyword-5"]["search_volume"] == 30
    assert result["keyword-11"]["search_volume"] == 30


def test_partial_metrics_are_preserved_when_last_batch_is_rate_limited():
    calls = []

    def requester(url, _headers, _timeout):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        hints = query["hintKeywords"][0].split(",")
        calls.append(hints)
        if len(calls) == 6:
            return 429, b'{"status":429}'
        return 200, json.dumps({
            "keywordList": [{
                "relKeyword": hint,
                "monthlyPcQcCnt": 10,
                "monthlyMobileQcCnt": 20,
                "compIdx": "HIGH",
            } for hint in hints]
        }).encode("utf-8")

    provider = make_provider(requester, max_retries=0)

    with pytest.raises(NaverKeywordProviderError) as error:
        provider.fetch_metrics([f"keyword-{index}" for index in range(30)])

    partial = error.value.partial_metrics
    assert len(calls) == 6
    assert partial["keyword-0"]["search_volume"] == 30
    assert partial["keyword-24"]["metrics_available"] is True
    assert partial["keyword-25"]["search_volume"] is None
    assert partial["keyword-25"]["metrics_available"] is False


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_errors_do_not_retry(status):
    calls = []

    def requester(_url, _headers, _timeout):
        calls.append(1)
        return status, b'{"message":"unauthorized"}'

    with pytest.raises(NaverKeywordProviderError, match="authentication"):
        make_provider(requester, max_retries=2).fetch_metrics(["keyword"])
    assert len(calls) == 1


@pytest.mark.parametrize("status", [429, 500])
def test_rate_limit_and_server_errors_use_bounded_retry(status):
    calls = []

    def requester(_url, _headers, _timeout):
        calls.append(1)
        return status, b"{}"

    with pytest.raises(NaverKeywordProviderError, match="temporary"):
        make_provider(requester, max_retries=2).fetch_metrics(["keyword"])
    assert len(calls) == 3


def test_timeout_uses_bounded_retry():
    calls = []

    def requester(_url, _headers, _timeout):
        calls.append(1)
        raise TimeoutError("timed out")

    with pytest.raises(NaverKeywordProviderError, match="timed out"):
        make_provider(requester, max_retries=2).fetch_metrics(["keyword"])
    assert len(calls) == 3


def test_malformed_response_is_reported():
    def requester(_url, _headers, _timeout):
        return 200, b'{"unexpected": []}'

    with pytest.raises(NaverKeywordProviderError, match="keywordList"):
        make_provider(requester).fetch_metrics(["keyword"])


def test_empty_keywords_do_not_make_a_request():
    def requester(*_args):
        raise AssertionError("requester should not be called")

    assert make_provider(requester).fetch_metrics([]) == {}


def test_missing_credentials_fail_without_mock_fallback():
    with pytest.raises(ValueError, match="credentials are not configured"):
        NaverKeywordProvider(api_key="", secret_key="", customer_id="")


def test_mock_mode_always_selects_mock_provider(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("KEYWORD_PROVIDER", "naver")
    get_settings.cache_clear()

    try:
        assert isinstance(build_keyword_provider(), MockKeywordDataProvider)
    finally:
        get_settings.cache_clear()
