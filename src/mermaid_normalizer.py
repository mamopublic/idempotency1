import re

# The single canonical classDef body used for all diagrams.
# Fixing this eliminates color/style variance between iterations.
CANONICAL_CLASSDEF_BODY = "fill:#f9f,stroke:#333,stroke-width:2px"

# Mermaid keywords that cannot be node IDs
_MERMAID_KEYWORDS = frozenset({
    'graph', 'flowchart', 'td', 'lr', 'bt', 'rl', 'tb',
    'subgraph', 'end', 'classDef', 'class', 'click',
    'style', 'linkStyle', 'note', 'direction',
})


def _quote_labels_with_special_chars(line: str) -> str:
    """Wraps unquoted Mermaid node labels that contain special characters in double quotes.

    Examples:
        A[Green Light (60s)]         →  A["Green Light (60s)"]
        B{Decision (yes/no)}         →  B{"Decision (yes/no)"}

    Already-quoted labels are left untouched.
    Does NOT touch classDef/class/comment lines.
    """
    stripped = line.strip()
    if (stripped.startswith('classDef ')
            or (stripped.startswith('class ') and not stripped.startswith('classDef'))
            or stripped.startswith('%%')):
        return line

    SPECIAL = r'[(){},;:/]'
    line = re.sub(
        r'(\w+)\[([^"\[\]]*' + SPECIAL + r'[^"\[\]]*)\]',
        lambda m: f'{m.group(1)}["{m.group(2)}"]',
        line
    )
    line = re.sub(
        r'(\w+)\(([^"()]*' + SPECIAL + r'[^"()]*)\)',
        lambda m: f'{m.group(1)}("{m.group(2)}")',
        line
    )
    line = re.sub(
        r'(\w+)\{([^"{}]*' + SPECIAL + r'[^"{}]*)\}',
        lambda m: f'{m.group(1)}{{"{m.group(2)}"}}',
        line
    )
    return line


def _collect_inline_class_node_ids(line: str) -> list:
    """Extract node IDs from inline :::className annotations.

    Handles both:
      - NodeID[Label]:::className   (shape + annotation)
      - NodeID:::className          (bare node + annotation)
    Returns list of node IDs found.
    """
    ids = []
    # Pattern: word (node ID) optionally followed by a bracket/paren/brace shape,
    # then optional whitespace, then :::
    pattern = r'\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\]|\([^)]*\)|\{[^}]*\})?\s*(?=:::[\w])'
    for m in re.finditer(pattern, line):
        candidate = m.group(1)
        if candidate.lower() not in _MERMAID_KEYWORDS:
            ids.append(candidate)
    return ids


def normalize_mermaid_code(code: str) -> str:
    """Normalizes Mermaid code to remove cosmetic/stylistic LLM variance.

    Normalizations applied:
    1. Normalize diagram type header: 'flowchart TD/LR' → 'graph TD/LR'
    2. Rename any classDef to canonical 'nodeStyle' with fixed body (eliminates color variance)
    3. Deduplicate multiple classDef lines (keep only one)
    4. Strip inline :::className annotations; record node IDs affected
    5. Collect node IDs from `class` application lines; drop those lines
    6. Emit a single canonical `class node1,node2,... nodeStyle` line (sorted)
    7. Normalize reserved-word node ID substitutes (EndNode → endNode, END → endNode)
    8. Auto-quote node labels that contain special characters (parens, braces, etc.)
    """

    lines = code.splitlines()

    # Pre-pass: collect all classDef class names (to normalize class-application lines)
    classdef_names = []
    for line in lines:
        m = re.match(r'\s*classDef\s+(\S+)', line)
        if m:
            name = m.group(1)
            if name not in classdef_names:
                classdef_names.append(name)

    # Main pass
    normalized_lines = []
    seen_canonical_classdef = False
    styled_node_ids = set()

    for line in lines:
        stripped = line.strip()

        # 1. Normalize diagram type header
        line = re.sub(r'\bflowchart\s+TD\b', 'graph TD', line)
        line = re.sub(r'\bflowchart\s+LR\b', 'graph LR', line)

        # 2+3. classDef → canonical, deduplicated
        if re.match(r'\s*classDef\s+\S+', line):
            indent = re.match(r'(\s*)', line).group(1)
            if not seen_canonical_classdef:
                normalized_lines.append(f"{indent}classDef nodeStyle {CANONICAL_CLASSDEF_BODY}")
                seen_canonical_classdef = True
            # drop duplicate classDef lines
            continue

        # 5. Collect and drop `class` application lines
        if stripped.startswith('class ') and not stripped.startswith('classDef'):
            m2 = re.match(r'\s*class\s+([\w,\s]+)\s+\S+', line)
            if m2:
                for node_id in m2.group(1).split(','):
                    nid = node_id.strip()
                    if nid:
                        # Apply reserved-word normalization to collected IDs too
                        nid = re.sub(r'^EndNode$', 'endNode', nid)
                        nid = re.sub(r'^END$', 'endNode', nid)
                        styled_node_ids.add(nid)
            continue

        # 4. Strip :::className annotations; capture affected node IDs first
        if ':::' in line:
            for nid in _collect_inline_class_node_ids(line):
                styled_node_ids.add(nid)
            line = re.sub(r':::[\w]+', '', line)

        # 7. Reserved-word substitutes
        line = re.sub(r'\bEndNode\b', 'endNode', line)
        line = re.sub(r'\bEND\b', 'endNode', line)

        # 8. Auto-quote labels with special characters
        line = _quote_labels_with_special_chars(line)

        normalized_lines.append(line)

    # Strip trailing blank lines, then append the canonical class application
    while normalized_lines and normalized_lines[-1].strip() == '':
        normalized_lines.pop()

    if styled_node_ids:
        sorted_ids = ','.join(sorted(styled_node_ids))
        normalized_lines.append(f"    class {sorted_ids} nodeStyle")

    return '\n'.join(normalized_lines)
