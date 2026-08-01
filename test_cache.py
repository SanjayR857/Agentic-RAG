import sys
from langcache import LangCache
import time

from backend.app.core.config import settings

api_key = settings.LANGCACHE_API_KEY
server_url = settings.LANGCACHE_SERVER_URL
cache_id = settings.LANGCACHE_CACHE_ID

prompt = "What is semantic caching test?"
response_text = "This is a test response for semantic caching."

try:
    with LangCache(server_url=server_url, cache_id=cache_id, api_key=api_key) as lc:
        print("Saving to cache...")
        save_res = lc.set(prompt=prompt, response=response_text)
        print("Save response:", type(save_res), save_res)
        
        time.sleep(1)
        
        print("Searching cache...")
        search_res = lc.search(prompt=prompt)
        print("Search response:", type(search_res), search_res)
        if isinstance(search_res, list):
            for i, x in enumerate(search_res):
                print(f"Hit {i}:", type(x), x)
                if hasattr(x, 'response'):
                    print(f"Hit {i} response attr:", getattr(x, 'response'))
        
        print("\n\nNow testing my service logic...")
        from backend.app.services.langcache_service import langcache_service
        ans = langcache_service.search_cache(prompt)
        print("My service returned:", type(ans), repr(ans))

except Exception as e:
    print("Error:", e)
