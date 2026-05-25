from pathlib import Path


def test_stoner_and_projected_hf_import_independently():
    import tdbg_scf.projected_hf.solver  # noqa: F401
    import tdbg_scf.stoner.solver  # noqa: F401


def test_stoner_and_projected_hf_do_not_import_each_other_statically():
    root = Path(__file__).resolve().parents[1] / "tdbg_scf"
    for path in (root / "stoner").glob("*.py"):
        assert "projected_hf" not in path.read_text(encoding="utf-8")
    for path in (root / "projected_hf").glob("*.py"):
        assert "stoner" not in path.read_text(encoding="utf-8")
