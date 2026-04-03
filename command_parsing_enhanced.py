"""
command_parsing_enhanced.py
===========================
Flexible command parser for face recognition surveillance system.

Supports positional commands:
  "detect second person right to User1"
  "find third person on left of Alice"
  "identify first person right of Bob"
  "could you take a photo of the 3rd person to my left"
  "locate our chief guest — he is 2 to the left of Raj"
  ... as well as single-person: "detect User1"
"""

import re

# ── Keyword banks ────────────────────────────────────────────────────────────

ACTION_WORDS = {
    "detect", "find", "identify", "scan", "show", "look",
    "search", "locate", "check", "spot", "see", "get",
    "who", "what", "is", "are", "there",
    "take", "grab", "capture", "snap", "photo", "picture",
    "point", "highlight", "can", "could", "would", "please",
}

# Filler words spoken naturally before the actual command.
# These are stripped first so the name extractor never sees them.
# e.g. "hello I want you to detect Aditya" → only "detect Aditya" is parsed
FILLER_WORDS = {
    "hello", "hi", "hey", "okay", "ok", "alright", "right",
    "so", "now", "well", "yes", "yeah", "yep", "please",
    "i", "want", "you", "me", "us", "we", "they", "them",
    "need", "like", "do", "did", "will", "go", "make",
    "tell", "ask", "let", "help", "try",
    "today", "currently", "quickly", "just", "also",
}

RIGHT_WORDS = {
    "right", "right-side", "rightside", "rightof", "rhs",
}

LEFT_WORDS = {
    "left", "left-side", "leftside", "leftof", "lhs",
}

# ── Position keyword → ordinal number ────────────────────────────────────────
POSITION_WORDS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    # bare digit strings — so "3 to the left" is treated as position 3 (change 13)
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    # natural aliases
    "next": 1, "nearest": 1,
}

CONNECTOR_WORDS = {
    "to", "of", "the", "a", "an", "on", "at", "in", "is", "are",
    "who", "beside", "next", "near", "adjacent", "side", "person",
    "someone", "anybody", "anyone", "standing", "sitting", "by",
    "from", "for", "me", "us", "them", "their", "and",
    # spoken filler words — filter these out during name extraction (change 12)
    # e.g. "could you find him", "locate our chief guest Mr Mohan"
    "him", "her", "his", "hers", "it", "this", "that",
    "just", "with", "our", "my", "your", "we",
    "today", "guest", "chief", "sir", "madam",
    "mr", "ms", "mrs", "dr", "here", "now", "okay", "ok",
}

# Noise for name extraction — exclude POSITION_WORDS so they don't eat names
ALL_NOISE_WORDS = (ACTION_WORDS | CONNECTOR_WORDS | RIGHT_WORDS | LEFT_WORDS
                   | set(POSITION_WORDS.keys()))


class CommandParser:
    """
    Parses free-form surveillance commands into structured data.

    Returns dict with keys:
      valid, reference_person, direction, position, mode, raw_command
    """

    def parse(self, command: str) -> dict:
        if not command or not command.strip():
            return self._error("Empty command")

        raw = command.strip()

        # ── 0. Strip leading filler words from spoken input ──────────────────
        # e.g. "hello I want you to detect Aditya"
        #   → "detect Aditya"
        # This is done on a lowered word list; the original-case raw is kept
        # for name extraction later.
        stripped_words = raw.split()
        while stripped_words:
            w_low = stripped_words[0].strip(".,!?;:").lower()
            if w_low in FILLER_WORDS:
                stripped_words.pop(0)
            else:
                break
        # Rebuild the working string after filler removal
        working = " ".join(stripped_words)
        if not working:
            return self._error("Command contained only filler words")

        text  = working.lower()
        words = text.split()

        # ── 1. Extract position word (first / second / third …) ─────────────
        position  = 1  # default
        pos_index = -1
        for i, w in enumerate(words):
            clean = w.strip(".,!?;:")
            if clean in POSITION_WORDS:
                position  = POSITION_WORDS[clean]
                pos_index = i
                break

        # ── 2. Detect direction ──────────────────────────────────────────────
        direction = None
        dir_index = -1
        for i, w in enumerate(words):
            clean = w.strip(".,!?;:")
            if clean in RIGHT_WORDS:
                direction = "right"
                dir_index = i
                break
            if clean in LEFT_WORDS:
                direction = "left"
                dir_index = i
                break

        # ── 3. Single-person mode (no direction) ─────────────────────────────
        if direction is None:
            orig_words = working.split()
            name = self._extract_name_smart(orig_words, 0, len(orig_words))
            if not name:
                return self._error(
                    "Could not identify a person name. "
                    "Examples: 'detect Alice' or 'find second person right to Bob'"
                )
            return {
                "valid":            True,
                "reference_person": name,
                "direction":        None,
                "position":         1,
                "raw_command":      raw,
                "mode":             "single",
            }

        # ── 4. Directional mode — extract reference person name ───────────────
        orig_words = working.split()
        # Prefer tokens after direction word
        name = self._extract_name_smart(orig_words, dir_index + 1, len(orig_words))
        if not name:
            name = self._extract_name_smart(orig_words, 0, dir_index)

        if not name:
            return self._error(
                "Could not identify a person name. "
                "Example: 'find second person to the right of Alice'"
            )

        return {
            "valid":            True,
            "reference_person": name,
            "direction":        direction,
            "position":         position,
            "raw_command":      raw,
            "mode":             "directional",
        }

    def _extract_name(self, words: list, start: int, end: int) -> str:
        """Original extractor — kept for backward compatibility."""
        name_tokens = []
        for w in words[start:end]:
            clean = w.strip(".,!?;:").lower()
            if clean and clean not in ALL_NOISE_WORDS and len(clean) > 1:
                name_tokens.append(w.strip(".,!?;:"))
        return " ".join(name_tokens) if name_tokens else ""

    def _extract_name_smart(self, words: list, start: int, end: int) -> str:
        """
        Smarter name extractor used after filler stripping.

        Strategy:
          1. Collect all non-noise tokens in the slice.
          2. Among those, strongly prefer tokens that start with a capital letter
             (proper nouns / names) — these are almost always the person name.
          3. If multiple capitalised tokens exist, take the last contiguous run
             (e.g. "detect Mr John Doe" → "Mr John Doe" → prefer "John Doe").
          4. Fall back to the original extractor result if no capitals found.

        This means "hello want you detect Aditya" after filler-stripping becomes
        "detect Aditya" and _extract_name_smart returns "Aditya" rather than
        "want you Aditya".
        """
        # Step 1: collect all non-noise tokens
        candidates = []
        for w in words[start:end]:
            clean = w.strip(".,!?;:").lower()
            if clean and clean not in ALL_NOISE_WORDS and len(clean) > 1:
                candidates.append(w.strip(".,!?;:"))

        if not candidates:
            return ""

        # Step 2: prefer capitalised tokens (proper nouns)
        capitalised = [t for t in candidates if t[0].isupper()]
        if capitalised:
            # Take the last capitalised token(s) — handles "Mr John Doe right of Raj"
            # where we want only the tokens after the connector keywords
            # Find the last run of capitalised tokens
            last_run = [capitalised[-1]]
            idx = candidates.index(capitalised[-1])
            # Expand left while the preceding candidate is also capitalised
            while idx > 0 and candidates[idx-1][0].isupper():
                idx -= 1
                last_run.insert(0, candidates[idx])
            return " ".join(last_run)

        # Step 3: fallback — return all non-noise tokens
        return " ".join(candidates)

    def format_feedback(self, result: dict) -> str:
        if not result.get("valid"):
            return f"Invalid: {result.get('error', 'Unknown error')}"

        name      = result["reference_person"]
        direction = result.get("direction")
        position  = result.get("position", 1)
        ordinal   = {1:"1st", 2:"2nd", 3:"3rd"}.get(position, f"{position}th")

        if direction is None:
            return f"Detecting '{name}' 🎯"

        arrow = "→" if direction == "right" else "←"
        return (f"Looking for {ordinal} person to the {direction.upper()} of "
                f"'{name}' {arrow}")

    @staticmethod
    def _error(msg: str) -> dict:
        return {
            "valid": False,
            "error": msg,
            "reference_person": None,
            "direction": None,
            "position": 1,
            "raw_command": "",
        }
