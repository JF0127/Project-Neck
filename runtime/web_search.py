"""Small local web-search tool backend for DeepSeek function calls."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .datetime_tool import CurrentDateTime
from .logging_utils import log

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class WebSearchError(RuntimeError):
    """A web search could not be completed."""


class WebSearch:
    """Search the web through Tavily and return compact text for the model."""

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        current_datetime: CurrentDateTime | None = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get("TAVILY_API_KEY", "")).strip()
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        self.max_results = int(max_results)
        self.current_datetime = current_datetime or CurrentDateTime()

    def _query_with_absolute_date(self, query: str) -> str:
        temporal_terms = ("今天", "今日", "目前", "当前", "最近")
        if not any(term in query for term in temporal_terms):
            return query
        current = self.current_datetime.get_current_datetime()
        year, month, day = (int(part) for part in current["date"].split("-"))
        absolute_date = f"{year}年{month}月{day}日"
        if "今天" in query or "今日" in query:
            return query.replace("今天", absolute_date).replace("今日", absolute_date)
        return f"{query} {absolute_date}"

    def search(self, query: str) -> str:
        clean_query = self._query_with_absolute_date(query.strip())
        if not clean_query:
            raise WebSearchError("web search query must not be empty")
        if not self.api_key:
            raise WebSearchError("TAVILY_API_KEY is required for web search")

        log(f"[web-search] query: {clean_query}")
        payload = json.dumps(
            {
                "api_key": self.api_key,
                "query": clean_query,
                "search_depth": "basic",
                "max_results": self.max_results,
                "include_answer": False,
                "include_raw_content": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            TAVILY_SEARCH_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20.0) as response:
                document: Any = json.load(response)
        except urllib.error.HTTPError as exc:
            raise WebSearchError(f"Tavily returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise WebSearchError(f"Tavily request failed: {type(exc).__name__}") from exc

        results = document.get("results", []) if isinstance(document, dict) else []
        lines: list[str] = []
        for index, result in enumerate(results, 1):
            if not isinstance(result, dict):
                continue
            title = str(result.get("title", "")).strip()
            url = str(result.get("url", "")).strip()
            content = str(result.get("content", "")).strip()
            if content:
                content = content[:1000]
            lines.append(
                f"[{index}] {title or 'Untitled'}\nURL: {url}\n摘要: {content}"
            )

        log("[web-search] completed")
        if not lines:
            return "未搜索到相关结果。"
        return "\n\n".join(lines)
