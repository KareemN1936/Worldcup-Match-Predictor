import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from config import FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL, RAW_API_JSON_DIR


class ApiClient:
    def __init__(
        self,
        base_url: str = FOOTBALL_DATA_BASE_URL,
        api_key: str = FOOTBALL_DATA_API_KEY,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-Auth-Token": self.api_key})

    def get(self, endpoint: str, params: dict[str, Any] | None = None, retries: int = 3) -> dict[str, Any] | None:
        if not self.api_key:
            print("FOOTBALL_DATA_API_KEY is missing. Skipping API request.")
            return None

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}

        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429 and attempt < retries:
                    wait_seconds = 10 * attempt
                    print(f"Rate limit hit. Waiting {wait_seconds} seconds before retrying {endpoint}.")
                    time.sleep(wait_seconds)
                    continue

                if response.status_code >= 500 and attempt < retries:
                    wait_seconds = 3 * attempt
                    print(f"Server error {response.status_code}. Retrying {endpoint} in {wait_seconds} seconds.")
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                return response.json()
            except requests.RequestException as error:
                print(f"API request failed for {endpoint} on attempt {attempt}/{retries}: {error}")
                if attempt < retries:
                    time.sleep(3 * attempt)

        print(f"Skipping {endpoint}; all retry attempts failed.")
        return None

    def save_json(self, data: Any, filename: str) -> Path:
        RAW_API_JSON_DIR.mkdir(parents=True, exist_ok=True)
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        if not safe_filename.endswith(".json"):
            safe_filename = f"{safe_filename}.json"
        path = RAW_API_JSON_DIR / safe_filename
        payload = {
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "data": data,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
