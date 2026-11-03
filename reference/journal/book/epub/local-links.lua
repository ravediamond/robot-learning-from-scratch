-- Links to repo files or other chapters do not resolve inside an e-reader: keep the text, drop the link.
function Link(el)
  local t = el.target
  if t:match('^https?://') or t:match('^#') or t:match('^mailto:') then return el end
  return el.content
end
