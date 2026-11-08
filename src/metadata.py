"""Conservative, explicit circuit and power-unit annotations.

Unknown teams/circuits stay unknown. Circuit character is a descriptive heuristic,
not an FIA classification. See README sources for the 2026 supplier changes.
"""

TEAM_COLORS = {
    "mercedes": "#27F4D2",
    "ferrari": "#E80020",
    "red_bull": "#3671C6",
    "mclaren": "#FF8000",
    "aston_martin": "#229971",
    "alpine": "#FF87BC",
    "williams": "#64C4FF",
    "haas": "#B6BABD",
    "rb": "#6692FF",
    "audi": "#F50537",
    "sauber": "#52E252",
    "cadillac": "#888888",
}


def circuit_type(circuit: str) -> str:
    """Return a coarse layout label, without estimating unknown venue properties."""
    if circuit in {"monaco", "baku", "marina_bay", "jeddah", "vegas", "miami", "albert_park"}:
        return "street"
    if circuit in {"monza", "spa", "silverstone", "suzuka", "jeddah"}:
        return "high_speed"
    if circuit in {"hungaroring", "zandvoort", "catalunya", "imola"}:
        return "technical"
    if circuit in {
        "bahrain",
        "americas",
        "red_bull_ring",
        "interlagos",
        "rodriguez",
        "yas_marina",
        "losail",
        "shanghai",
        "sepang",
        "sochi",
        "hockenheimring",
        "nurburgring",
        "portimao",
        "mugello",
        "ricard",
        "villeneuve",
    }:
        return "permanent"
    return "unknown"


def power_unit(team: str, year: int) -> str:
    """Map 2018–2026 constructor identities to engine suppliers."""
    if team in {"mercedes", "williams", "force_india", "racing_point"}:
        return "mercedes"
    if team == "mclaren":
        return "mercedes" if year >= 2021 else "renault"
    if team in {"renault", "alpine"}:
        return "mercedes" if year >= 2026 and team == "alpine" else "renault"
    if team == "aston_martin":
        return "honda" if year >= 2026 else "mercedes"
    if team in {"ferrari", "haas", "cadillac"}:
        return "ferrari"
    if team in {"sauber", "alfa", "audi"}:
        return "audi" if year >= 2026 else "ferrari"
    if team in {"red_bull", "rb", "racing_bulls", "alphatauri", "toro_rosso"}:
        if year >= 2026:
            return "red_bull_ford"
        if team == "red_bull" and year == 2018:
            return "renault"
        return "honda"
    return "unknown"
