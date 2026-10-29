from owlapy.owl_ontology import SyncOntology
from owlapy.owl_reasoner import SyncReasoner
from owlapy.class_expression import OWLClass
from owlapy.owl_axiom import OWLSubClassOfAxiom
from owlapy.static_funcs import startJVM
from owlapy import owl_expression_to_manchester


# Avvia la JVM
startJVM()

# Carica l'ontologia LUBM
ontology = SyncOntology("univ-bench-inst_no_individuals.ttl")
print
# Inizializza HermiT
reasoner = SyncReasoner(ontology, reasoner="HermiT")

# Namespace dell'ontologia LUBM originale
# ns = "http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#"
ns = "http://swat.cse.lehigh.edu/onto/univ-bench.owl#"

# Definisci le classi
student = OWLClass(ns + "Student")
person = OWLClass(ns + "Person")

# Costruisci la query: Student ⊑ Person
query = OWLSubClassOfAxiom(student, person)

# Verifica l'inferenza
if not reasoner.has_consistent_ontology():
    raise ValueError("Ontologia inconsistente")

result = reasoner.is_entailed(query)

def verify_subclass(reasoner, subclass_iri, superclass_iri):
    print(f"verifico relazione tra : {superclass_iri} e {subclass_iri}")
    sub = OWLClass(subclass_iri)
    sup = OWLClass(superclass_iri)

    axiom = OWLSubClassOfAxiom(sub, sup)
    return reasoner.is_entailed(axiom)


result = verify_subclass(
    reasoner,
    ns + "Student",
    ns + "Person"
)
# True
print("Student è sottoclasse di Person?", result)


def get_justifications(reasoner, axiom, n_max=10):
    """
    Verifica se un assioma è entailed e recupera
    le relative justifications.
    """
    entailed = reasoner.is_entailed(axiom)

    if not entailed:
        return {
            "entailed": False,
            "justifications": []
        }

    justifications = reasoner.create_axiom_justifications(
        axiom,
        n_max_justifications=n_max ## The maximum number of justifications to generate. Defaults to 10.
    )

    return {
        "entailed": True,
        "justifications": justifications
    }

if result: 
    axiom = OWLSubClassOfAxiom(student, person)
    print("assioma creato: ",axiom)
    result_j = get_justifications(reasoner, axiom)
    print("ottengo justification di student e person")
    for i, justification in enumerate(result_j["justifications"], 1):
        print(f"\nJustification {i}:")
        for ax in justification:
            print(ax)

# ### justification: 
# insieme minimo di assiomi sufficienti a implicare la conclusione, non una dimostrazione ordinata contenente ogni passaggio inferenziale.
# justifications = reasoner.create_axiom_justifications(
#     query,
#     n_max_justifications=5
# )

# for i, justification in enumerate(justifications, 1):
#     print(f"Justification {i}:")
#     for axiom in justification:
#         print(axiom)

### justification in dl syntax

from owlapy import owl_expression_to_dl
from owlapy.owl_axiom import OWLEquivalentClassesAxiom

axiom = OWLSubClassOfAxiom(student, person)


def axiom_to_dl(axiom):
    # owl_expression_to_dl rende solo class expression, non assiomi:
    # ogni tipo di assioma va ricostruito a partire dalle sue espressioni.
    if isinstance(axiom, OWLEquivalentClassesAxiom):
        expressions = [
            owl_expression_to_dl(expr)
            for expr in axiom.class_expressions()
        ]
        return " ≡ ".join(expressions)

    if isinstance(axiom, OWLSubClassOfAxiom):
        return (f"{owl_expression_to_dl(axiom.get_sub_class())} ⊑ "
                f"{owl_expression_to_dl(axiom.get_super_class())}")

    raise NotImplementedError(type(axiom).__name__)

print("QUERY IN DL: \n", axiom_to_dl(axiom))

def render_justifications(justifications):
    return [
        [
            owl_expression_to_manchester(axiom)
            for axiom in justification
        ]
        for justification in justifications
    ]
if result:
    for i, justification in enumerate(result_j["justifications"], 1):
        print(f"\nJUSTIFICATION {i} IN DL:")
        for ax in justification:
            print("  ", axiom_to_dl(ax))
            print(f"\nJUSTIFICATION {i} IN manchester syntax:")
            print("  ", owl_expression_to_manchester(ax))

### justification in manchester syntax: 



## creazione di assiomi in ttl 

from owlapy.owl_ontology import SyncOntology
from owlapy.iri import IRI

# Crea una nuova ontologia
onto = SyncOntology(
    IRI.create("http://example.org/justification_student_query"),
    load=False  # crea un'ontologia vuota invece di cercare un file con quel nome
)

# Inserisci l'assioma ottenuto
onto.add_axiom(axiom)

# Oppure in Turtle
onto.save(
    "justification_student_person.ttl",
    document_format="turtle"
)