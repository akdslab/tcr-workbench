"""Skill folders meet the packaging rules of the Biohub MCP host that serves them.

Biohub MCP copies each skills/<name>/ folder byte for byte from one commit of this
repository and refuses files it cannot serve, so a violation here only surfaces late,
in that repository's CI. These checks mirror the rules this repository can test on
its own; the host also pins hashes and keeps evals/ unserved, which it enforces itself.
"""
import json
from pathlib import Path
import re
import stat

import pytest

SKILLS = Path(__file__).resolve().parents[1] / "skills"
NAMES = sorted(p.name for p in SKILLS.iterdir()
               if p.is_dir() and not p.name.endswith("-workspace"))
# Served as text; anything else, including .json, .yaml and LICENSE, is served as binary.
TEXT_SUFFIXES = {".md", ".txt", ".py", ".csv"}
# The names skill-creator gives eval inputs and results; refused outside evals/.
RESERVED = ("benchmark", "feedback", "fixtures/", "-workspace/")
# Clients reading through the read_resource tool fail near 60 KB.
MAX_FILE_BYTES = 50_000
# Never committed, and the host copies from a commit rather than from this disk.
LOCAL_ONLY = {".DS_Store", "__pycache__"}
NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
EVAL_KEYS = {"id", "prompt", "expected_output", "files", "expectations"}


def files(skill):
    root = SKILLS / skill
    return [p for p in sorted(root.rglob("*"))
            if not LOCAL_ONLY.intersection(p.relative_to(root).parts) and not p.is_dir()]


def served(skill):
    root = SKILLS / skill
    return [p for p in files(skill) if p.relative_to(root).parts[0] != "evals"]


def test_skills_exist():
    assert NAMES == ["tcr-workbench", "tcr-workbench-setup"]


@pytest.mark.parametrize("skill", NAMES)
def test_served_files_are_plain_text_the_host_accepts(skill):
    root = SKILLS / skill
    for path in served(skill):
        relative = path.relative_to(root).as_posix()
        assert path.suffix in TEXT_SUFFIXES, f"{relative} would be served as binary"
        assert not any(name in relative for name in RESERVED), f"{relative} uses a reserved name"
        size = len(path.read_bytes())
        assert 0 < size < MAX_FILE_BYTES, f"{relative} is {size} bytes"
        path.read_text(encoding="utf-8")


@pytest.mark.parametrize("skill", NAMES)
def test_skill_folder_has_no_links(skill):
    for path in files(skill):
        metadata = path.lstat()
        assert not stat.S_ISLNK(metadata.st_mode), f"{path} is a symlink"
        assert metadata.st_nlink == 1, f"{path} is a hardlink"


@pytest.mark.parametrize("skill", NAMES)
def test_frontmatter_has_single_line_name_and_description(skill):
    assert NAME.match(skill)
    text = (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    assert "\n".join(lines[end + 1:]).strip(), "SKILL.md needs a body"
    header = lines[1:end]
    for key in ("name", "description"):
        values = [line[len(key) + 1:].strip() for line in header if line.startswith(f"{key}:")]
        assert len(values) == 1, f"{key}: must appear exactly once"
        # A YAML block (`>` or `|`) is served as the literal indicator.
        assert values[0] and values[0][0] not in ">|", f"{key}: must be a single line"
    assert [line for line in header if line.startswith("name:")] == [f"name: {skill}"]


@pytest.mark.parametrize("skill", NAMES)
def test_evals_have_the_skill_creator_shape(skill):
    evals = json.loads((SKILLS / skill / "evals/evals.json").read_text(encoding="utf-8"))
    assert evals["skill_name"] == skill
    cases = evals["evals"]
    assert cases
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert EVAL_KEYS <= case.keys(), f"case {case.get('id')} lacks {EVAL_KEYS - case.keys()}"
        assert isinstance(case["files"], list) and case["expectations"]
