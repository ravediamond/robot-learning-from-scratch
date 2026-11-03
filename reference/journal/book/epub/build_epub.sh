#!/usr/bin/env bash
# Build the book as an EPUB: journal/book/epub/road-to-robotics-book.epub
# Usage: bash journal/book/epub/build_epub.sh   (needs pandoc >= 3.0 for --split-level)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; book="$(dirname "$here")"
cd "$book"
chapters=$(ls [0-9]*.md | LC_ALL=C sort)
extras="exercises/interview/opener.md exercises/interview/faults_cards.md exercises/interview/levine_questions.md exercises/interview/pushback_world_models.md exercises/coding/README.md"
date=$(git log -1 --format=%cs -- . 2>/dev/null || date +%F)
rev=$(git log -1 --format=%h -- . 2>/dev/null || echo local)
pandoc $chapters $extras \
  -f markdown-tex_math_dollars-raw_html+pipe_tables \
  -t epub3 \
  --toc --toc-depth=2 --split-level=1 \
  --lua-filter "$here/local-links.lua" \
  --css "$here/epub.css" \
  --metadata title="Evaluating Robot Learning on Cheap Hardware" \
  --metadata subtitle="A research engineer's method for a one-person lab — edition $date ($rev)" \
  --metadata author="Ravindu Somawansa" \
  --metadata lang=en \
  --metadata date="$date" \
  -o "$here/road-to-robotics-book.epub"
echo "built $here/road-to-robotics-book.epub ($(du -h "$here/road-to-robotics-book.epub" | cut -f1), $(echo $chapters | wc -w) chapters + $(echo $extras | wc -w) exercise files, $rev)"
