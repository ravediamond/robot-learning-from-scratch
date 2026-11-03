# The book as an EPUB

`bash journal/book/epub/build_epub.sh` builds `road-to-robotics-book.epub` here (git-ignored) from the chapter files
in file-name order plus five exercise files, with `local-links.lua` (links to repo files flattened to text) and
`epub.css`. Published as the `book-latest` release of the repository, replaced on every rebuild (WS-063).

- pandoc **3.6.4** (static linux-arm64 release in `~/.local/bin`, installed by the owner 2026-10-30).
- epubcheck: not installed (a downloaded Java jar; not attempted after the pandoc download was refused).
- Checks run after each build: nav top-level entries = chapter + exercise files (40), `<table` count = markdown table
  separator lines in the sources (52), no `class="math"` (dollar amounts must stay text).
