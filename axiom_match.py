"""
Normalizzazione degli assiomi citati dal modello per il confronto con il groundtruth.

Serve perche' il confronto per stringa esatta misura la formattazione, non il contenuto:
il template gold prodotto da subclass_justification usa un placeholder ('...') al posto
delle restrizioni anonime, mentre il modello tipicamente le espande per intero.
Le due scritture denotano lo STESSO assioma e devono combaciare.
"""
import re

PREFIXED = re.compile(r"(?:owl|rdf|rdfs|xsd):")


def _strip_bracket_blocks(text):
    """Rimuove i blocchi [ ... ] (restrizioni anonime), gestendo l'annidamento."""
    out, depth = [], 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _named_classes(text):
    """Nomi locali ':Nome' che non sono termini di vocabolario (owl:, rdf:, ...)."""
    names = []
    for m in re.finditer(r"(?<![A-Za-z]):([A-Za-z_][\w-]*)", text):
        if PREFIXED.search(text[max(0, m.start() - 5):m.start() + 1]):
            continue
        names.append(m.group(1))
    return names


def canonical_axiom(axiom):
    """
    Riduce un assioma alla sua forma canonica confrontabile:
      ('subClassOf', sub, sup)                      per rdfs:subClassOf
      ('equivalentIntersection', cls, frozenset)    per owl:equivalentClass + intersectionOf
      ('declaration', cls)                          per ':X rdf:type owl:Class' (rumore)
      None                                          se non riconosciuto
    """
    a = " ".join(str(axiom).split())

    if re.search(r"\brdf:type\s+owl:Class\b", a) and "equivalentClass" not in a and "subClassOf" not in a:
        names = _named_classes(a)
        return ("declaration", names[0]) if names else None

    if "rdfs:subClassOf" in a:
        left, right = a.split("rdfs:subClassOf", 1)
        ln, rn = _named_classes(left), _named_classes(_strip_bracket_blocks(right))
        if ln and rn:
            return ("subClassOf", ln[0], rn[0])
        return None

    if "equivalentClass" in a:
        left, right = a.split("equivalentClass", 1)
        ln = _named_classes(left)
        if not ln:
            return None
        m = re.search(r"intersectionOf\s*\(", right)
        if m:
            inner = right[m.end():]
            members = frozenset(_named_classes(_strip_bracket_blocks(inner)))
        else:
            members = frozenset(_named_classes(_strip_bracket_blocks(right)))
        return ("equivalentIntersection", ln[0], members)

    return None


def canonical_set(axioms, drop_declarations=True):
    """Insieme canonico degli assiomi; separa le righe non riconosciute."""
    canon, unparsed, declarations = set(), [], []
    for a in axioms or []:
        c = canonical_axiom(a)
        if c is None:
            unparsed.append(a)
        elif c[0] == "declaration":
            declarations.append(a)
            if not drop_declarations:
                canon.add(c)
        else:
            canon.add(c)
    return canon, unparsed, declarations


def parse_axioms_block(block):
    """
    Dal blocco <axioms> ricava la lista di assiomi: una riga = un assioma, ma le righe
    con parentesi quadre/tonde non bilanciate vengono unite alla successiva (capita che
    il modello vada a capo dentro una restrizione anonima spezzando un singolo assioma).
    """
    if block is None:
        return None

    raw_lines = []
    for line in block.strip().splitlines():
        line = line.strip()
        line = re.sub(r"^[-*•]\s*", "", line)     # bullet eventuali
        line = re.sub(r"^\d+[.)]\s*", "", line)         # numerazione eventuale
        line = line.strip(" .;")
        if not line or line.upper() == "NONE":
            continue
        raw_lines.append(line)

    axioms, buf, depth = [], "", 0
    for line in raw_lines:
        buf = f"{buf} {line}".strip() if buf else line
        depth += line.count("[") - line.count("]") + line.count("(") - line.count(")")
        if depth <= 0:
            axioms.append(buf)
            buf, depth = "", 0
    if buf:
        axioms.append(buf)
    return axioms
