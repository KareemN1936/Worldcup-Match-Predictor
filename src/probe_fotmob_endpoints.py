from urllib.parse import urlencode

import requests

from fotmob_utils import FOTMOB_BASE_URL, FOTMOB_HEADERS, save_json, FOTMOB_RAW_DIR


PROBES = [
    ("matches", {"date": "20260618"}),
    ("matches", {"date": "20260618", "timezone": "UTC"}),
    ("matchDetails", {"matchId": "4667820"}),
    ("teams", {"id": "8634"}),
    ("teams", {"teamId": "8634"}),
    ("searchData", {"term": "Barcelona"}),
    ("search", {"term": "Barcelona"}),
]


def main() -> None:
    print(f"Probing FotMob base URL: {FOTMOB_BASE_URL}")
    results = []
    for endpoint, params in PROBES:
        url = f"{FOTMOB_BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, params=params, headers=FOTMOB_HEADERS, timeout=20)
            content_type = response.headers.get("content-type", "")
            preview = response.text[:180].replace("\n", " ")
            record = {
                "endpoint": endpoint,
                "params": params,
                "url": f"{url}?{urlencode(params)}",
                "status_code": response.status_code,
                "content_type": content_type,
                "preview": preview,
            }
            print(f"{response.status_code} {record['url']} [{content_type}]")
            if response.ok and "json" in content_type:
                record["json_keys"] = sorted(response.json().keys()) if isinstance(response.json(), dict) else []
            results.append(record)
        except Exception as error:
            print(f"ERROR {endpoint}: {error}")
            results.append({"endpoint": endpoint, "params": params, "error": str(error)})

    output_path = FOTMOB_RAW_DIR / "endpoint_probe.json"
    save_json(results, output_path)
    print(f"Probe results saved: {output_path}")


if __name__ == "__main__":
    main()
