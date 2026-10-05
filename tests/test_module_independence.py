"""The public numerical package has a single full-band Hartree path."""
import ast
from pathlib import Path


def test_full_hartree_and_stoner_import_without_retired_projection_modules():
    import tdbg_scf
    import tdbg_scf.stoner.solver  # noqa: F401

    root = Path(tdbg_scf.__file__).resolve().parent
    assert not (root / "solver_projected.py").exists()
    assert not (root / "projected_hf").exists()
    assert not any("Projected" in name or "projected" in name for name in tdbg_scf.__all__)


def test_numerical_modules_have_no_projection_imports():
    root = Path(__file__).resolve().parents[1] / "tdbg_scf"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any("solver_projected" in module or "projected_hf" in module for module in modules), path
