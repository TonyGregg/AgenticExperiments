"""
Cache utility for LLM API responses.
"""
import diskcache as dc
import hashlib
import json
from typing import Any, Callable, Optional


class LLMCache:
    """Cache manager for LLM API responses."""

    def __init__(self, cache_dir: str = "./data/llm_cache"):
        self.cache = dc.Cache(cache_dir)

    def _generate_key(self, model_name: str, query: str, **kwargs) -> str:
        key_data = {
            "model": model_name,
            "query": query,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def cached_api_call(
            self,
            model_name: str,
            query: str,
            api_function: Callable,
            force_refresh: bool = False,
            **api_kwargs
    ) -> Any:
        if not force_refresh:
            cached = self.get(model_name, query, **api_kwargs)
            if cached is not None:
                print(f"✓ Cache hit for {model_name}")
                return cached

        print(f"✗ Cache miss for {model_name} - calling API...")
        response = api_function(query, **api_kwargs)
        self.set(model_name, query, response, **api_kwargs)
        return response

    def get(self, model_name: str, query: str, **kwargs) -> Optional[Any]:
        key = self._generate_key(model_name, query, **kwargs)
        return self.cache.get(key)

    def set(self, model_name: str, query: str, response: Any, **kwargs) -> None:
        key = self._generate_key(model_name, query, **kwargs)
        self.cache.set(key, response)