## justification 
una justification è un insieme minimale di assiomi di un'ontologia sufficiente a dimostrare una determinata conseguenza logica (entailment).

Consideriamo un'ontologia $(\mathcal{O})$, costituita da un insieme di assiomi, e una conseguenza logica $(\alpha)$.

Una justification è un sottoinsieme $(J \subseteq \mathcal{O}$) che soddisfa due proprietà:

1. Sufficienza (entailment)
$[
J \models \alpha
$]

Gli assiomi contenuti in $(J)$ sono sufficienti per inferire $(\alpha)$.

2. Minimalità rispetto all'inclusione
$[
\forall J' \subsetneq J,\quad J' \not\models \alpha
]$

Nessun sottoinsieme proprio di \(J\) è sufficiente a dimostrare \(\alpha\).

Quindi:
Una justification è un insieme minimale di assiomi che, considerati congiuntamente, implicano una determinata conclusione.

Attenzione: minimale non significa necessariamente di cardinalità minima. Una justification può contenere cinque assiomi anche quando ne esiste un'altra composta da tre.

MUPS	Insieme minimale di assiomi che rende insoddisfacibile una specifica classe<br>
MIPS	Insieme minimale di assiomi che rende incoerente la TBox, nel senso di avere almeno una classe nominata insoddisfacibile