from __future__ import annotations

from typing import Any
import re

from action_estimation import parse_instruction_actions


GOAL_KEYWORDS = [
    "目标点",
    "出口",
    "房间",
    "垃圾桶",
    "垃圾",
    "座位",
    "桌子",
    "椅子",
    "厕所",
    "休息",
    "找",
    "去",
    "拿",
    "door",
    "exit",
    "room",
    "seat",
    "table",
    "chair",
    "toilet",
    "rest",
    "trash",
    "garbage",
    "goal",
    "find",
    "go to",
    "pick up",
]


def analyze_instruction(instruction: str) -> dict[str, Any]:
    """Analyze whether an instruction is goal-level or low-level action language."""
    text = instruction or ""
    instruction_actions = parse_instruction_actions(text)
    goal_keywords = _find_goal_keywords(text)

    has_actions = bool(instruction_actions)
    has_goal = bool(goal_keywords)

    if has_goal and has_actions:
        instruction_type = "goal_level_with_action_hint"
        analysis_note = (
            "Instruction contains goal-level task language and low-level action hints; "
            "treat it primarily as goal-level unless explicit action labels are needed."
        )
    elif has_goal:
        instruction_type = "goal_level"
        analysis_note = (
            "Instruction describes a goal or task rather than a complete low-level action sequence."
        )
    elif has_actions:
        instruction_type = "low_level_action"
        analysis_note = (
            "Instruction appears to contain explicit low-level navigation action commands."
        )
    else:
        instruction_type = "unknown"
        analysis_note = "No clear goal keywords or low-level action commands were detected."

    return {
        "instruction_type": instruction_type,
        "instruction_actions": instruction_actions,
        "goal_keywords": goal_keywords,
        "goal_description": text.strip(),
        "analysis_note": analysis_note,
    }


def _find_goal_keywords(instruction: str) -> list[str]:
    lowered = instruction.lower()
    matches: list[tuple[int, str]] = []
    for keyword in GOAL_KEYWORDS:
        for start in _find_keyword_positions(lowered, keyword.lower()):
            matches.append((start, instruction[start : start + len(keyword)]))

    matches.sort(key=lambda item: item[0])
    deduped: list[str] = []
    seen: set[str] = set()
    for _, matched_text in matches:
        normalized = matched_text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(matched_text)
    return deduped


def _find_keyword_positions(text: str, keyword: str) -> list[int]:
    if keyword.isascii():
        pattern = r"(?<![A-Za-z])" + re.escape(keyword) + r"(?![A-Za-z])"
        return [match.start() for match in re.finditer(pattern, text)]

    positions: list[int] = []
    start = 0
    while True:
        index = text.find(keyword, start)
        if index == -1:
            break
        positions.append(index)
        start = index + len(keyword)
    return positions
