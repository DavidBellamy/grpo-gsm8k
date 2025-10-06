import re
from typing import Literal

BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUM_RE = re.compile(r"-?\d+(\.\d+)?")
ANSWER_RE = re.compile(r"(?im)^\s*answer\s*:\s*(.+)\s*$")


def extract_boxed(text: str) -> str | None:
    m = BOX_RE.findall(text)
    return m[-1].strip() if m else None


def extract_answer_colon(text: str) -> str | None:
    """
    Extracts the last 'ANSWER: <...>' line (case-insensitive), if present.
    """
    m = ANSWER_RE.findall(text)
    return m[-1].strip() if m else None


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return m.group(0) if m else s


def exact_match(
    pred: str,
    gold_solution: str,
    parser: Literal["auto", "boxed", "answer"] = "auto",
) -> int:
    """
    Compare a predicted answer to the GSM8K gold solution.
    - Gold: parsed from the text after '####'.
    - Pred parsing is controlled by `parser`:
        - 'boxed'  -> use \\boxed{...}
        - 'answer' -> use 'ANSWER: <...>'
        - 'auto'   -> try boxed, then answer
    Returns 1 for match, 0 otherwise.
    """
    gold = gold_solution.split("####")[-1].strip()

    pred_str: str | None
    if parser == "boxed":
        pred_str = extract_boxed(pred)
    elif parser == "answer":
        pred_str = extract_answer_colon(pred)
    else:  # auto
        pred_str = extract_boxed(pred) or extract_answer_colon(pred)

    if pred_str is None:
        return 0
    return int(normalize_number(pred_str) == normalize_number(gold))


def _ngram_repetition_ratio(tokens: list[int], n: int = 3) -> float:
    """
    Simple repetition proxy: unique n-grams / total n-grams (lower means more repetition).
    Returns 1.0 when sequence is too short to form at least two n-grams.
    """
    if len(tokens) < n + 1:
        return 1.0
    total = len(tokens) - n + 1
    seen = {tuple(tokens[i : i + n]) for i in range(total)}
    return len(seen) / total


def reward_from_text(
    pred: str,
    gold_solution: str,
    parser: Literal["auto", "boxed", "answer"] = "auto",
) -> float:
    """
    Compute a scalar reward indicating whether a predicted answer matches a gold solution.
    This function relies on `exact_match` to compare `pred` and `gold_solution` after
    parsing/extracting the final answer, then converts the result to a float.
    Args:
        pred: The model's predicted answer text.
        gold_solution: The reference (ground-truth) answer text. Extracted using '####'.
        parser: Strategy for extracting the model's final answer before comparison.
            One of:
            - "boxed": Extract the content inside a LaTeX \\boxed{...}.
            - "answer": Use case-insensitive "ANSWER:<numeric>" markers to locate the final answer.
            - "auto": Try to extract the answer using both strategies.
    Returns:
        float: 1.0 if the answers match under the chosen parsing strategy, 0.0 otherwise.
    Example:
        >>> reward_from_text("The answer is \\\\boxed{42}.", "#### 42", parser="boxed")
        1.0
    """
    return float(exact_match(pred, gold_solution, parser))
