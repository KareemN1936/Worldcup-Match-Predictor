import pandas as pd


FIFA_TO_COUNTRY_CODE = {
    "ALG": "DZ",
    "ARG": "AR",
    "AUS": "AU",
    "AUT": "AT",
    "BEL": "BE",
    "BIH": "BA",
    "BRA": "BR",
    "CAN": "CA",
    "CPV": "CV",
    "CIV": "CI",
    "COL": "CO",
    "CRC": "CR",
    "CRO": "HR",
    "CZE": "CZ",
    "COD": "CD",
    "ECU": "EC",
    "EGY": "EG",
    "ENG": "GB",
    "FRA": "FR",
    "GER": "DE",
    "GHA": "GH",
    "HAI": "HT",
    "IRN": "IR",
    "IRQ": "IQ",
    "JPN": "JP",
    "JOR": "JO",
    "KOR": "KR",
    "MAR": "MA",
    "MEX": "MX",
    "NED": "NL",
    "NZL": "NZ",
    "NOR": "NO",
    "PAN": "PA",
    "PAR": "PY",
    "POR": "PT",
    "QAT": "QA",
    "KSA": "SA",
    "SCO": "GB",
    "SEN": "SN",
    "RSA": "ZA",
    "ESP": "ES",
    "SWE": "SE",
    "SUI": "CH",
    "TUN": "TN",
    "TUR": "TR",
    "USA": "US",
    "URU": "UY",
    "UZB": "UZ",
}


def country_code_to_emoji(country_code: str | None) -> str:
    if not country_code or not isinstance(country_code, str) or len(country_code) != 2:
        return "🏳️"
    code = country_code.upper()
    return "".join(chr(ord(char) + 127397) for char in code if "A" <= char <= "Z")


def fifa_code_to_emoji(fifa_code: str | None) -> str:
    return country_code_to_emoji(FIFA_TO_COUNTRY_CODE.get(str(fifa_code).upper()))


def build_flag_lookup(teams: pd.DataFrame) -> dict[str, str]:
    if teams.empty or "team_name" not in teams.columns:
        return {}

    lookup: dict[str, str] = {}
    for _, row in teams.iterrows():
        team = str(row.get("team_name", "")).strip()
        if not team:
            continue
        flag = "🏳️"
        if pd.notna(row.get("flag_url")):
            flag = str(row.get("flag_url"))
        elif pd.notna(row.get("fifa_code")):
            flag = fifa_code_to_emoji(str(row.get("fifa_code")))
        lookup[team] = flag
    return lookup


def flag_for_team(team: str, lookup: dict[str, str]) -> str:
    return lookup.get(str(team).strip(), "🏳️")
