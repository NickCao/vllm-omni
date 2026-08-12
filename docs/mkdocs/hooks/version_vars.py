# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Substitutes version placeholder tokens with the pinned upstream vLLM
release, so install instructions scattered across the docs can be bumped
in one place (the `extra.vllm_version` value in mkdocs.yml) instead of
hunting down every hardcoded occurrence.

Supported tokens:
  - `{{vllm_version}}` -> full release, e.g. "0.27.0"
  - `{{vllm_minor}}`   -> major.minor, e.g. "0.27"

This can't be a normal `on_page_markdown`/`on_page_content` mkdocs hook.
Most usages sit inside fenced ```bash blocks in install docs that are only
ever pulled into a real page (gpu.md) through a pymdownx.snippets `--8<--`
include, and both the snippet expansion and the Pygments syntax
highlighting happen *inside* the single `markdown.Markdown().convert()`
call that a page hook can only see the before/after of:

  - `on_page_markdown` sees gpu.md's raw text, before the `--8<--`
    reference is expanded, so the token (which lives in the *included*
    file) isn't present in the text yet.
  - `on_page_content` sees the final HTML, but by then Pygments has
    already tokenized the fenced code block into separate <span>s (e.g.
    `{{` and `}}` end up in different spans), so the token is no longer a
    contiguous string to search-and-replace.

So instead `on_config` registers a real Python-Markdown preprocessor
(`_VersionVarsExtension`), passing a live instance straight into
`config.markdown_extensions` (Python-Markdown accepts already-constructed
Extension objects there, so this needs no import-path/sys.path setup).
Preprocessors run in descending priority order on plain text lines, before
any block-level parsing. pymdownx.snippets registers its preprocessor at
priority 32 and pymdownx.superfences pulls fenced blocks out for
highlighting at priority 25, so priority 28 runs after snippets have
expanded includes but before code fences are extracted and highlighted -
the one window where the text is both fully assembled and still plain.

`on_post_build` is a defense-in-depth check: it strips HTML tags before
searching the built site for a leftover token, since a plain substring
search would miss a token that Pygments (or any other treeprocessor)
fragmented across tags - the same blind spot that let the original,
broken on_page_content approach ship silently.
"""

import re
from pathlib import Path

from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError

_TOKEN_RE = re.compile(r"\{\{\s*vllm_(?:version|minor)\s*\}\}")
_TAG_RE = re.compile(r"<[^>]+>")

# Below pymdownx.snippets (32), above pymdownx.superfences' fenced-code
# extraction (25).
_PREPROCESSOR_PRIORITY = 28


class _VersionVarsPreprocessor(Preprocessor):
    def __init__(self, md, version: str, minor: str) -> None:
        super().__init__(md)
        self.version = version
        self.minor = minor

    def run(self, lines: list[str]) -> list[str]:
        return [line.replace("{{vllm_minor}}", self.minor).replace("{{vllm_version}}", self.version) for line in lines]


class _VersionVarsExtension(Extension):
    def __init__(self, version: str) -> None:
        self.version = version
        self.minor = ".".join(version.split(".")[:2])
        super().__init__()

    def extendMarkdown(self, md) -> None:
        md.preprocessors.register(
            _VersionVarsPreprocessor(md, self.version, self.minor),
            "vllm_version_vars",
            _PREPROCESSOR_PRIORITY,
        )


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    version = config.extra.get("vllm_version")
    if version:
        config.markdown_extensions.append(_VersionVarsExtension(version))
    return config


def on_post_build(*, config: MkDocsConfig) -> None:
    site_dir = Path(config.site_dir)
    leaked = sorted(
        str(path.relative_to(site_dir))
        for path in site_dir.rglob("*.html")
        if _TOKEN_RE.search(_TAG_RE.sub("", path.read_text(encoding="utf-8")))
    )
    if leaked:
        raise PluginError(
            "Unresolved {{vllm_version}}/{{vllm_minor}} token(s) found in built pages "
            f"(hooks/version_vars.py failed to substitute them): {', '.join(leaked)}"
        )
