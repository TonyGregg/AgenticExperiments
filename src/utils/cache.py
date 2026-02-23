"""
Cache utility for LLM API responses.
"""
import diskcache as dc
import hashlib
import json
from typing import Any, Callable, Optional


from src.utils.config import Config


class LLMCache:
    """Cache manager for LLM API responses."""

    def __init__(self, cache_dir: str = "./data/llm_cache"):
        self.cache = dc.Cache(cache_dir)

    def _generate_key(
            self,
            model_name: str,
            query: Any,
            use_full_context: bool = False,  # New parameter
            **kwargs
    ) -> str:
        """
        Generate cache key.

        use_full_context=False: Cache by last user message only (chatbot use case)
        use_full_context=True:  Cache by entire message history (agentic use case)
        """
        if isinstance(query, list):
            if use_full_context:
                # Use entire message list as key (for agentic tasks)
                cache_query = json.dumps(query, sort_keys=True)
            else:
                # Use only last user message (for chatbot)
                cache_query = ""
                for msg in reversed(query):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        cache_query = msg.get("content", "")
                        break
        else:
            cache_query = query

        key_data = {
            "model": model_name,
            "query": cache_query,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
            self,
            model_name: str,
            query: Any,
            use_full_context: bool = False,
            **kwargs
    ) -> Optional[Any]:
        key = self._generate_key(model_name, query, use_full_context, **kwargs)
        return self.cache.get(key)

    def set(
            self,
            model_name: str,
            query: Any,
            response: Any,
            use_full_context: bool = False,
            **kwargs
    ) -> None:
        key = self._generate_key(model_name, query, use_full_context, **kwargs)
        self.cache.set(key, response)

    def cached_api_call(
            self,
            model_name: str,
            query: Any,
            api_function: Callable,
            force_refresh: bool = False,
            use_full_context: bool = False,  # New parameter
            **api_kwargs
    ) -> Any:
        if not force_refresh:
            cached = self.get(model_name, query, use_full_context, **api_kwargs)
            if cached is not None:
                print(f"✓ Cache hit for [{model_name}]")
                return cached

        print(f"✗ Cache miss for [{model_name}] - calling API...")
        response = api_function(query, **api_kwargs)
        self.set(model_name, query, response, use_full_context, **api_kwargs)
        return response

    def get_cache_size(self):
        return {
            "cache_size": len(self.cache) if hasattr(self.cache, '__len__') else "unknown",
            "cache_directory": str(Config.CACHE_DIR)
        }

    def clear(self):
        self.cache.clear()