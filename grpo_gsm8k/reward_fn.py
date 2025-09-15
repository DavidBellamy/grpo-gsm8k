import re

BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUM_RE = re.compile(r"-?\d+(\.\d+)?")


def extract_boxed(text: str) -> str | None:
    m = BOX_RE.findall(text)
    return m[-1].strip() if m else None


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return m.group(0) if m else s


def exact_match(pred: str, gold_solution: str) -> int:
    """
    GSM8K gold_solution contains a rationale and '#### final_number' at end.
    We parse the number after '####'.
    """
    gold = gold_solution.split("####")[-1].strip()
    pred_box = extract_boxed(pred)
    if pred_box is None:
        return 0
    return int(normalize_number(pred_box) == normalize_number(gold))


def reward_from_text(pred: str, gold_solution: str) -> float:
    return float(exact_match(pred, gold_solution))
