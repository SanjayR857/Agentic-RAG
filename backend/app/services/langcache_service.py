from langcache import LangCache
from backend.app.core.config import settings

class LangCacheService:
    def __init__(self):
        self.api_key = settings.LANGCACHE_API_KEY
        self.server_url = settings.LANGCACHE_SERVER_URL
        self.cache_id = settings.LANGCACHE_CACHE_ID

    def search_cache(self, prompt: str) -> str | None:
        """Search the cache for a semantically similar prompt and return the response."""
        try:
            with LangCache(
                server_url=self.server_url,
                cache_id=self.cache_id,
                api_key=self.api_key,
            ) as lang_cache:
                search_response = lang_cache.search(prompt=prompt)
                
                # The SDK returns a SearchResponse object with a .data attribute containing CacheEntry objects
                if search_response and hasattr(search_response, "data") and search_response.data:
                    first_hit = search_response.data[0]
                    if hasattr(first_hit, "response"):
                        return first_hit.response
                
                return None
        except Exception as e:
            print(f"Error searching LangCache: {e}")
            return None

    def save_to_cache(self, prompt: str, response: str) -> bool:
        """Save a prompt and response pair to the cache."""
        try:
            with LangCache(
                server_url=self.server_url,
                cache_id=self.cache_id,
                api_key=self.api_key,
            ) as lang_cache:
                save_response = lang_cache.set(
                    prompt=prompt,
                    response=response
                )
                return True
        except Exception as e:
            print(f"Error saving to LangCache: {e}")
            return False

langcache_service = LangCacheService()
