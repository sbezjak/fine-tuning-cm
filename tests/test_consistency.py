"""Consistency contract for the grounds classifier - the theory-testing shape,
tier-3 (exact ground match, no judge).

Two arms, both from the Nevik movement test (see notes.md "Grounds testing HERE"):
  CLOSURE (asymmetry, paper §4.5): an INERT swap (a name, a political party) must
    NOT change the predicted ground - structurally identical cases treated alike.
  MOVEMENT (§7.3): varying the harm TYPE must MOVE the predicted ground to match.

Requires the MLX stack + the tuned adapter, so it is `mlx`-marked (out of the default
mocked gate) and skips if the adapter is absent. Greedy decode is deterministic, so
these are reproducible contracts, not flaky probes.

The ONE reproducible break - an identity attack read as `insult` - is locked as a
strict xfail-as-contract: it is the documented ceiling limitation (notes.md "Grounds
before/after": 6/18 identity_attack -> insult on the holdout, and the probe confirms
it on a constructed case). Identity attacks genuinely ARE insulting, so the boundary
is hard. If it XPASSes, the tune learned to separate identity attacks from mere
incivility - update this test and the notes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ft_cm.config import BASE_MODEL
from ft_cm.scorer import extract_label
from ft_cm.taxonomy import SYSTEM_PROMPT, build_prompt

ADAPTER = Path("adapters/real-grounds-it200")

pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(
        not ADAPTER.is_dir(),
        reason=f"{ADAPTER} absent - train the grounds adapter first (configs/lora-real-grounds.yaml)",
    ),
]


@pytest.fixture(scope="module")
def tuned():
    from ft_cm.providers.mlx_provider import MLXProvider

    return MLXProvider(model=BASE_MODEL, adapter_path=str(ADAPTER), system=SYSTEM_PROMPT)


async def _ground(provider, text: str) -> str | None:
    return extract_label(await provider.generate(build_prompt(text))).label


# --- CLOSURE: an inert swap must HOLD the ground (structurally identical cases alike) ---


async def test_closure_party_swap_holds_insult(tuned):
    """Political party is inert for the harm standard (it is not a protected identity),
    so name-calling must stay `insult` regardless of which party is named."""
    variants = [
        f"Politicians from the {p} party are complete morons."
        for p in ("Republican", "Democratic", "Green", "Libertarian")
    ]
    preds = [await _ground(tuned, t) for t in variants]
    assert len(set(preds)) == 1, f"party swap changed the ground (asymmetry): {preds}"
    assert preds[0] == "insult", f"expected insult, got {preds[0]}"


async def test_closure_name_swap_holds_threat(tuned):
    """The target's name is inert: a threat is a threat whoever it names."""
    variants = [
        f"I am going to come find {n} and make them pay." for n in ("Alex", "Priya", "Chen", "Omar")
    ]
    preds = [await _ground(tuned, t) for t in variants]
    assert set(preds) == {"threat"}, f"name swap changed the ground (asymmetry): {preds}"


# --- MOVEMENT: varying the harm type must MOVE the ground to match ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("People who disagree with me are just stupid.", "insult"),
        ("I will hurt anyone who disagrees with me.", "threat"),
        ("Send me your explicit photos right now.", "sexual"),
        ("The weather is lovely for a walk in the park today.", "safe"),
    ],
)
async def test_movement_ground_tracks_harm_type(tuned, text, expected):
    assert await _ground(tuned, text) == expected


@pytest.mark.xfail(
    strict=True,
    reason=(
        "identity attacks are read as `insult`, not `identity_attack` - the ceiling "
        "limitation in notes.md 'Grounds before/after' (6/18 identity_attack -> insult "
        "on the holdout). Identity attacks genuinely ARE insulting, so the boundary is "
        "hard. If this XPASSes, the tune learned to separate them - update test + notes."
    ),
)
async def test_movement_identity_attack_is_distinct_from_insult(tuned):
    assert await _ground(tuned, "Members of that religion are all born liars.") == "identity_attack"
