"""Pure validation and spatial helpers for Gemma 4 crowd analysis."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping


class InvalidGemmaResponse(ValueError):
    """Raised when Gemma's structured response cannot be used safely."""


ALLOWED_RISK_LEVELS = {"low", "moderate", "high"}
ALLOWED_ACTIONS = {"continue_monitoring", "notify_safety_operator"}
MAX_PEOPLE_PER_FRAME = 300
MAX_REASON_LENGTH = 500


def _decode_payload(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidGemmaResponse("Gemma response was not valid JSON") from exc
        if isinstance(decoded, Mapping):
            return dict(decoded)
    raise InvalidGemmaResponse("Gemma response must be a JSON object")


def parse_gemma_response(response) -> dict:
    """Extract one structured analysis from Gemma tool arguments or JSON text."""
    calls = getattr(response, "function_calls", None) or []
    for call in calls:
        if getattr(call, "name", None) == "report_crowd_analysis":
            return _decode_payload(getattr(call, "args", None))
    return _decode_payload(getattr(response, "text", None))


def validate_analysis(payload) -> dict:
    """Validate all fields before results are used for counts or advisories."""
    data = _decode_payload(payload)
    required = {
        "people_count", "persons", "risk_level", "crowding_detected",
        "crowd_observation", "recommended_action", "reason",
    }
    missing = required - data.keys()
    if missing:
        raise InvalidGemmaResponse("Gemma analysis is missing required fields")

    people_count = data["people_count"]
    if isinstance(people_count, bool) or not isinstance(people_count, int):
        raise InvalidGemmaResponse("people_count must be an integer")
    if not 0 <= people_count <= MAX_PEOPLE_PER_FRAME:
        raise InvalidGemmaResponse("people_count is outside the accepted range")

    raw_persons = data["persons"]
    if not isinstance(raw_persons, list) or len(raw_persons) > MAX_PEOPLE_PER_FRAME:
        raise InvalidGemmaResponse("persons must be a bounded list")
    persons = []
    for person in raw_persons:
        if not isinstance(person, Mapping):
            raise InvalidGemmaResponse("each person detection must be an object")
        coords = person.get("box_2d")
        if not isinstance(coords, list) or len(coords) != 4:
            raise InvalidGemmaResponse("box_2d must contain [ymin, xmin, ymax, xmax]")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in coords):
            raise InvalidGemmaResponse("box coordinates must be numbers")
        if not all(math.isfinite(value) and int(value) == value for value in coords):
            raise InvalidGemmaResponse("box coordinates must be finite integers")
        ymin, xmin, ymax, xmax = (int(value) for value in coords)
        if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
            raise InvalidGemmaResponse("box coordinates are out of range or unordered")
        persons.append({"box_2d": [ymin, xmin, ymax, xmax]})
    if people_count != len(persons):
        raise InvalidGemmaResponse("people_count must match the number of validated boxes")

    risk_level = data["risk_level"]
    if not isinstance(risk_level, str) or risk_level.strip().lower() not in ALLOWED_RISK_LEVELS:
        raise InvalidGemmaResponse("risk_level is not allowed")
    if not isinstance(data["crowding_detected"], bool):
        raise InvalidGemmaResponse("crowding_detected must be boolean")
    recommended_action = data["recommended_action"]
    if not isinstance(recommended_action, str) or recommended_action not in ALLOWED_ACTIONS:
        raise InvalidGemmaResponse("recommended_action is not allowed")

    result = dict(data)
    result.update({
        "people_count": people_count,
        "persons": persons,
        "risk_level": risk_level.strip().lower(),
        "crowding_detected": data["crowding_detected"],
        "recommended_action": recommended_action,
    })
    for field in ("crowd_observation", "reason"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise InvalidGemmaResponse(f"{field} must be non-empty text")
        result[field] = value.strip()[:MAX_REASON_LENGTH]
    return result


def normalized_box_to_pixels(box_2d, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert Gemma's [ymin,xmin,ymax,xmax] 0–1000 box to pixel xyxy."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    ymin, xmin, ymax, xmax = box_2d
    x1 = min(int(xmin * width / 1000), width - 1)
    y1 = min(int(ymin * height / 1000), height - 1)
    x2 = min(max(int(xmax * width / 1000), x1 + 1), width)
    y2 = min(max(int(ymax * height / 1000), y1 + 1), height)
    return x1, y1, x2, y2


def calculate_grid(persons, width: int, height: int, rows: int, cols: int) -> list[list[int]]:
    """Count person-box centers in a rows × cols grid."""
    if min(width, height, rows, cols) <= 0:
        raise ValueError("image dimensions and grid dimensions must be positive")
    counts = [[0 for _ in range(cols)] for _ in range(rows)]
    for person in persons:
        x1, y1, x2, y2 = normalized_box_to_pixels(person["box_2d"], width, height)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        col = min(center_x * cols // width, cols - 1)
        row = min(center_y * rows // height, rows - 1)
        counts[row][col] += 1
    return counts


def result_is_fresh(captured_at: float, now: float, max_age: float) -> bool:
    """Return true only when a result timestamp is valid and within its age limit."""
    age = now - captured_at
    return 0 <= age <= max_age
