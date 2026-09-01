"""Guards that a before/after run is actually measuring the ADAPTER, not silently
the base model twice. See the verification gap noted during the walkthrough: when
the baseline is saturated (before == after == 1.000), the numbers alone cannot tell
a working adapter from one that never loaded, so a false 0 delta looks identical to
a real one. These tests make the silent no-op loud.
"""

import pytest

from ft_cm.eval import _assert_adapter_present


@pytest.mark.mocked
def test_missing_adapter_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _assert_adapter_present(str(tmp_path / "does-not-exist"))


@pytest.mark.mocked
def test_dir_without_adapter_config_raises(tmp_path):
    # a directory that exists but holds no adapter_config.json is not a trained adapter
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        _assert_adapter_present(str(tmp_path / "empty"))


@pytest.mark.mocked
def test_real_smoke_adapter_passes():
    # the committed smoke adapter dir has adapter_config.json, so the guard is silent
    _assert_adapter_present("adapters/smoke")


@pytest.mark.mlx
async def test_adapter_actually_changes_output():
    """The inert-adapter case the file check cannot catch: load the base and the
    tuned model for real and assert they DIFFER on at least one probe. If they are
    byte-identical everywhere, the adapter is absent or inert and this fails loudly.
    Requires the MLX stack + the downloaded base model + a trained adapters/smoke.
    """
    from ft_cm.config import ADAPTER_PATH, BASE_MODEL
    from ft_cm.providers.mlx_provider import MLXProvider
    from ft_cm.taxonomy import SYSTEM_PROMPT, build_prompt

    base = MLXProvider(model=BASE_MODEL, adapter_path=None, system=SYSTEM_PROMPT)
    tuned = MLXProvider(model=BASE_MODEL, adapter_path=str(ADAPTER_PATH), system=SYSTEM_PROMPT)

    # Probe choice is BASE-DEPENDENT (finding, 2026-09-01). A probe only exposes the
    # adapter where base and tuned DISAGREE, i.e. in the base's blind spot. The old
    # probes ("...coffee at the library...") were chosen for the 0.5B, which over-
    # flagged them; the 1.5B base already gets them right, so base==tuned there and
    # the guard false-alarmed "inert" even though the before/after eval clearly moves
    # (evidence/hard-1.5b-before-after.json). These idiomatic-violence benign lines sit
    # in the adapter's STRONGEST learned direction (lean-safe on benign) and flip
    # base(unsafe)->tuned(safe) on BOTH 0.5B and 1.5B. Re-verify them on any base swap.
    probes = [
        "That final boss killed me at least twenty times before I finally beat it.",
        "My legs are dead after leg day, I can barely make it up the stairs.",
        "This summer heat is killing me, I cannot wait for autumn.",
    ]
    base_out = [await base.generate(build_prompt(p)) for p in probes]
    tuned_out = [await tuned.generate(build_prompt(p)) for p in probes]

    assert base_out != tuned_out, (
        "base and tuned produced identical output on every probe - the adapter is "
        "either not loading or inert, so a before/after delta would be meaningless"
    )
