from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_contract():
    headings = [
        line[3:] for line in (ROOT / "README.md").read_text().splitlines() if line.startswith("## ")
    ]
    assert headings == [
        "Problem",
        "Scope",
        "Quick start",
        "Usage",
        "Architecture",
        "Reproduction",
        "Results",
        "Testing",
        "Limitations",
        "Contributing",
        "License",
    ]


def test_public_text_has_no_em_dash():
    files = list(ROOT.glob("*.md"))
    for directory in ("src", "tests", "docs", "fixtures", "evidence"):
        files.extend(
            p for p in (ROOT / directory).rglob("*") if p.suffix in {".py", ".md", ".json"}
        )
    for path in files:
        assert chr(0x2014) not in path.read_text(), path
