"""
Minimal-justification builder per query di subClassOf sull'ontologia univ-bench.

Corregge tre problemi del codice originale (find_superclass_path / build_proof_chain):

1. Caso riflessivo (sub == sup) non gestito: la BFS parte cercando i *superclassi*
   di `start` e non controlla mai se start == end prima di iniziare, quindi
   "Is X a subclass of X?" (vero per definizione) non trova mai un path.

2. Le classi definite via owl:equivalentClass + intersectionOf (Employee,
   Director, Dean, Chair, Student, TeachingAssistant) NON hanno quasi mai un
   arco rdfs:subClassOf esplicito verso il loro "genitore concettuale": il
   codice originale, appoggiandosi solo a reasoner.super_classes(direct=True),
   etichetterebbe comunque l'arco come 'TBox (direct axiom)', cioè come se
   fosse un rdfs:subClassOf asserito — che è FALSO e fuorviante quando si deve
   giudicare se la spiegazione di un LLM cita l'assioma giusto. Es: Employee
   non ha alcun rdfs:subClassOf, la sua unica relazione con :Person viene
   dall'intersectionOf dentro owl:equivalentClass.

3. Il branch 'subClassOf_negative' eticheta l'assenza di path come
   "Reasoner (closed-world over the hierarchy)". È concettualmente sbagliato:
   la non-entailment di una subsumption è un risultato SOUND sotto OWA
   (si dimostra costruendo un modello in cui le due classi non si
   sovrappongono), non un'assunzione di closed-world. Vista la discussione
   fatta sul prompt, lasciare questa dicitura nella justification
   rinforzerebbe proprio l'errore concettuale che si vuole misurare nei modelli.

In più: per Director, che nell'ontologia è definito SOLO come
equivalentClass di una restrizione (∃headOf.Program) senza alcun named class
come membro dell'intersectionOf, non esiste alcun percorso verso :Person —
è un caso limite reale e interessante da tenere come test.
"""

from collections import deque
import rdflib
from rdflib import RDFS, RDF, OWL, Namespace
from rdflib.collection import Collection

UNIV = Namespace("http://swat.cse.lehigh.edu/onto/univ-bench.owl#")


def load_graph(ttl_path):
    g = rdflib.Graph()
    g.parse(ttl_path, format="turtle")
    return g


def local_name(uri):
    return str(uri).split("#")[-1]


def describe_restriction(g, bnode):
    """Da un blank node owl:Restriction estrae (proprietà, tipo, riempitore) leggibili."""
    prop = g.value(bnode, OWL.onProperty)
    for kind, label in [(OWL.someValuesFrom, "∃"), (OWL.allValuesFrom, "∀")]:
        filler = g.value(bnode, kind)
        if filler is not None:
            return f"{label}{local_name(prop)}.{local_name(filler)}"
    card = g.value(bnode, OWL.cardinality) or g.value(bnode, OWL.minCardinality)
    if card is not None:
        return f"card({local_name(prop)}) {card}"
    return f"restriction on {local_name(prop)}"


def build_named_subclass_edges(g):
    """
    Costruisce, per ogni classe con nome, la lista di archi verso superclassi
    NOMINATE, distinguendo la fonte assiomatica:
      - 'asserted'      : rdfs:subClassOf esplicito verso una classe con nome
      - 'equivalentClass': la classe è owl:equivalentClass di un
                            intersectionOf che contiene, tra i suoi membri,
                            una classe con nome (es. Employee -> Person)
    Ritorna: dict[str, list[dict]]  node -> [{'to':..., 'source':..., 'axiom':...}]
    """
    edges = {}

    def add_edge(sub, sup, source, axiom_text):
        edges.setdefault(sub, []).append(
            {"to": sup, "source": source, "axiom": axiom_text}
        )

    # 1) rdfs:subClassOf esplicito verso classi nominate (ignora restrizioni anonime)
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(o, rdflib.URIRef) and isinstance(s, rdflib.URIRef):
            sub, sup = local_name(s), local_name(o)
            add_edge(sub, sup, "asserted",
                     f":{sub} rdfs:subClassOf :{sup}")

    # 2) owl:equivalentClass -> intersectionOf: estrai i membri che sono classi
    #    nominate (superclassi implicite) e descrivi le restrizioni a parte
    for cls, _, eq in g.triples((None, OWL.equivalentClass, None)):
        if not isinstance(cls, rdflib.URIRef):
            continue
        cls_name = local_name(cls)
        inter = g.value(eq, OWL.intersectionOf)
        if inter is None:
            continue
        members = list(Collection(g, inter))
        restr_descriptions = []
        for m in members:
            if isinstance(m, rdflib.URIRef):
                sup_name = local_name(m)
                add_edge(cls_name, sup_name, "equivalentClass",
                         f":{cls_name} owl:equivalentClass "
                         f"[intersectionOf ( :{sup_name} ... )]")
            else:
                restr_descriptions.append(describe_restriction(g, m))
        # conserva le restrizioni come nota informativa sul nodo
        if restr_descriptions:
            edges.setdefault(cls_name, [])
            edges[cls_name + "__restrictions__"] = restr_descriptions

    return edges


def find_path(edges, start, end, skip=frozenset({"Thing", "Nothing"})):
    """BFS sugli archi verso superclassi nominate, con caso riflessivo gestito."""
    if start == end:
        return [{"node": start, "via": None, "axiom": None}]

    queue = deque([(start, [{"node": start, "via": None, "axiom": None}])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for e in edges.get(node, []):
            sup = e["to"]
            if sup in skip:
                continue
            new_path = path + [{"node": sup, "via": e["source"], "axiom": e["axiom"]}]
            if sup == end:
                return new_path
            if sup not in visited:
                visited.add(sup)
                queue.append((sup, new_path))
    return None


def build_subclass_justification(g, edges, sub, sup):
    """
    Ritorna un dict con:
      - answer: True / False
      - steps: lista human-readable
      - axioms: lista strutturata (per scoring automatico precision/recall
                contro le citazioni presenti nella explanation del modello)
      - caveat: eventuale nota (es. classe definita solo da restrizione anonima)
    """
    path = find_path(edges, sub, sup)
    result = {"answer": path is not None, "steps": [], "axioms": [], "caveat": None}

    if path is None:
        result["steps"].append(
            f"No subsumption chain found from :{sub} to :{sup} in the "
            f"hierarchy (via rdfs:subClassOf or owl:equivalentClass)."
        )
        result["steps"].append(
            "This is a sound non-entailment result under OWA: there exists "
            f"a model of the ontology in which :{sub} and :{sup} do not "
            "overlap, so ':Sub ⊑ :Sup' is not derivable — this is not a "
            "closed-world assumption."
        )
        restr = edges.get(sub + "__restrictions__")
        if restr:
            result["caveat"] = (
                f":{sub} is also defined by anonymous conditions ({', '.join(restr)}) "
                "that do not introduce additional named superclasses."
            )
        return result

    if len(path) == 1:
        result["steps"].append(f":{sub} ⊑ :{sub} by reflexivity of ⊑.")
        result["axioms"].append(f":{sub} rdfs:subClassOf :{sub} (reflexivity)")
        return result

    for i in range(1, len(path)):
        prev, cur = path[i - 1]["node"], path[i]["node"]
        source, axiom = path[i]["via"], path[i]["axiom"]
        if source == "asserted":
            result["steps"].append(f":{prev} ⊑ :{cur}  — {axiom}")
        else:
            restr = edges.get(prev + "__restrictions__")
            extra = f", with additional condition {restr}" if restr else ""
            result["steps"].append(
                f":{prev} ⊑ :{cur}  — derived from {axiom}{extra} "
                "(NOT an explicit rdfs:subClassOf)"
            )
        result["axioms"].append(axiom)

    if len(path) > 2:
        chain = " ⊑ ".join(f":{p['node']}" for p in path)
        result["steps"].append(f"By transitivity of ⊑: {chain}  ⟹  :{sub} ⊑ :{sup}")

    return result


if __name__ == "__main__":
    TTL = "univ-bench-inst.ttl"
    g = load_graph(TTL)
    edges = build_named_subclass_edges(g)

    test_cases = [
        ("Institute", "Lecturer"),      # atteso: False (esempio della trace incollata)
        ("University", "Organization"), # atteso: True, asserted, 1 hop
        ("Lecturer", "Person"),         # atteso: True, catena asserted multi-hop
        ("Employee", "Person"),         # atteso: True, ma via equivalentClass, non asserted!
        ("Director", "Person"),         # atteso: False (caso limite: solo restrizione anonima)
        ("Chair", "Faculty"),           # atteso: True, asserted (Chair subClassOf Professor subClassOf Faculty)
        ("University", "University"),  # atteso: True, riflessivo
    ]

    for sub, sup in test_cases:
        print("=" * 70)
        print(f"Is :{sub} a subClassOf :{sup} ?")
        res = build_subclass_justification(g, edges, sub, sup)
        print(f"-> answer: {res['answer']}")
        for s in res["steps"]:
            print(f"   {s}")
        if res["caveat"]:
            print(f"   [caveat] {res['caveat']}")
        print(f"   axioms (for scoring): {res['axioms']}")