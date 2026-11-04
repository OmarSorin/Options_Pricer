"""Keep the docs honest: run the README's example and check that its links point somewhere.

The README's Monte Carlo example is written as a doctest (lines starting with >>>), so this
test runs it and compares the output with what the README says. If the code changes and the
numbers drift, this fails instead of the README quietly going stale.
"""

import doctest
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def test_readme_examples_run_and_print_what_the_readme_says():
    text = README.read_text(encoding="utf-8")
    # doctest reads expected output until a blank line, so it would treat the closing ``` of
    # the code block as part of the output. Turn the fence lines into blank lines first.
    text = re.sub(r"^```.*$", "", text, flags=re.MULTILINE)
    parser = doctest.DocTestParser()
    test = parser.get_doctest(text, globs={}, name="README.md", filename=str(README), lineno=0)
    assert test.examples, "expected at least one >>> example in the README"
    # ELLIPSIS: "10.4205..." in the README matches the full float 10.420541193...
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    runner.run(test)
    assert runner.failures == 0, f"{runner.failures} README example(s) printed something else; see output above"


DOCS = [README, *sorted((ROOT / "docs").glob("*.md"))]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_point_to_existing_files(doc):
    # Markdown links [text](target) and HTML src/srcset="target"; skip web links and #anchors.
    text = doc.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:src|srcset)="([^"]+)"', text)
    local = [t.split("#")[0] for t in targets if not t.startswith(("http://", "https://", "#", "mailto:"))]
    missing = [t for t in local if t and not (doc.parent / t).exists()]
    assert not missing, f"broken links in {doc.name}: {missing}"
