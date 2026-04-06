# constants.py — shared position mappings used by data_prep.py and app.py

POSITION_MAP = {
    "GK":  "Goalkeeper",
    "DC":  "Centre-Back",
    "DR":  "Right Back",
    "DL":  "Left Back",
    "WBR": "Right Wing Back",
    "WBL": "Left Wing Back",
    "DMC": "Defensive Midfielder",
    "DMR": "Right Defensive Midfielder",
    "DML": "Left Defensive Midfielder",
    "MC":  "Central Midfielder",
    "MR":  "Right Midfielder",
    "ML":  "Left Midfielder",
    "AMC": "Attacking Midfielder",
    "AMR": "Right Attacking Midfielder",
    "AML": "Left Attacking Midfielder",
    "FW":  "Striker",
    "FWR": "Right Winger",
    "FWL": "Left Winger",
    "SS":  "Second Striker",
    "Sub": "Substitute",
}

GROUPED_POSITION_MAP = {
    "GK":  "Goalkeeper",
    "DC":  "Centre-Back",
    "DR":  "Right Defender",
    "WBR": "Right Defender",
    "DL":  "Left Defender",
    "WBL": "Left Defender",
    "DMC": "Defensive Midfielder",
    "DMR": "Defensive Midfielder",
    "DML": "Defensive Midfielder",
    "MC":  "Central Midfielder",
    "MR":  "Central Midfielder",
    "ML":  "Central Midfielder",
    "AMC": "Attacking Midfielder",
    "AMR": "Right Winger",
    "FWR": "Right Winger",
    "AML": "Left Winger",
    "FWL": "Left Winger",
    "FW":  "Striker",
    "SS":  "Striker",
    "Sub": "Substitute",
}

# Derived: group label → set of full position labels (no Substitute)
GROUP_TO_FULL_POSITIONS: dict[str, set[str]] = {}
for _code, _grp in GROUPED_POSITION_MAP.items():
    if _grp == "Substitute":
        continue
    _full = POSITION_MAP.get(_code, _code)
    GROUP_TO_FULL_POSITIONS.setdefault(_grp, set()).add(_full)
