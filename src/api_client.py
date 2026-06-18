import time
from typing import Any

import requests

from config import API_BASE_URL, API_KEY


class ApiClient:
    def __init__(self, base_url: str = API_BASE_URL, api_key: str = API_KEY, timeout: int = 30):
        if not base_url:
            raise ValueError("API_BASE_URL is not configured in .env")
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get(self, path: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(1, retries + 1):
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code < 500:
                response.raise_for_status()
                return response.json()
            if attempt == retries:
                response.raise_for_status()
            time.sleep(attempt)
        raise RuntimeError("unreachable")
