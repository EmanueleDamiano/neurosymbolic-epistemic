# neurosymbolic-epistemic

Pacchetto collaborativo degli ultimi esperimenti LUBM/owlapy, copiato il 25 settembre 2026. Gli originali non sono stati modificati.

## Pilot di revisione delle credenze

Aprire `belief_revision_lubm.ipynb` per una prima prova con Ollama locale e
`qwen2.5-coder:7b` (deve essere già installato). Il notebook esegue tre chiamate:
baseline, ritiro di un assioma necessario e aggiunta di un assioma irrilevante come
controllo. I due aggiornamenti partono dalla stessa risposta iniziale.

Il contesto è un frammento estratto e verificato da LUBM, composto solo da inclusioni
positive fra classi nominate; non è l'intera ontologia. L'oracolo di raggiungibilità
è completo per questo frammento e distingue entailment da non-entailment, senza
interpretare quest'ultimo come falsità. Si valutano risposta e supporto citato;
la revisione è interpretata condizionatamente a una baseline corretta e giustificata.

Ambiente minimo separato dalle dipendenze degli esperimenti storici, senza Java:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-belief-revision.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name epistemic-pilot --display-name "Python (epistemic pilot)"
```

In un editor Jupyter selezionare `Python (epistemic pilot)` ed eseguire tutte le celle.
La cella di configurazione permette di cambiare modello. Prompt, risposte grezze,
digest del modello, versione Ollama, opzioni, hash dell'ontologia e valutazioni
vengono salvati in `generated/belief_revision/<run_id>/` senza sovrascrivere i run.
Questo è un singolo episodio dimostrativo, non una stima delle capacità del modello.

### Seconda sezione: intera ontologia e supporti alternativi

La sezione `Esperimento 2 — Ontologia LUBM completa` dello stesso notebook è
autonoma e conserva il pilot precedente. Invia tutte le 319 triple di
`univ-bench-inst.ttl`, riserializzate senza perdita; negli aggiornamenti ritira
solo gli assiomi previsti. Le citazioni sono triple con URI completi, non ID artificiali.
HermiT verifica entailment e consistenza di ogni snapshot, e la sufficienza dei
supporti citati. Occorrono Java nel PATH e le dipendenze aggiuntive:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-belief-revision-full.txt
```

I due casi comprendono sei chiamate Ollama: il caso iniziale su tutta LUBM e un
caso con catena di tre inclusioni più una via alternativa sintetica documentata.
Nel secondo caso ritirare una sola via deve conservare la conclusione, ritirarle
entrambe deve farla abbandonare. I rami ripartono indipendentemente dalla baseline.
Il contesto Ollama è impostato a 32768 token, con controlli conservativi per evitare
troncamenti. Snapshot, prompt, risposte e valutazioni sono conservati in
`generated/belief_revision_full/<run_id>/`.

## Benchmark di grounding e ritiro su query positive

`subclass_grounding_benchmark.ipynb` implementa il protocollo a score fisso,
separato dal notebook esplorativo. Il numero di query positive di sottoclasse tratte da
`owlapy_generated_qa.json` è configurabile: `N_QUERIES = "all"` oppure un intero positivo.
Il dataset originale di 6 casi rimane in `benchmarks/subclass_grounding_v1/`.
Il nuovo `benchmarks/subclass_grounding_v2_all/` contiene tutte le 52 query ammissibili:
33 con prova esplicita da 1 passaggio, 15 da 2 e 4 da 3. Le prove ammesse
sono insiemi di inclusioni esplicite fra classi nominate; tutto il resto di LUBM
rimane nel contesto. HermiT verifica ogni snapshot prima dell'esecuzione.

Ogni query produce due chiamate: ontologia originale e intera ontologia con un singolo
assioma necessario rimosso, conservando la risposta originale nella cronologia.
HermiT conferma `entailed` prima e `not_entailed` dopo. Il modello restituisce solo
`answer` e `axioms`, con triple URI complete. L'evaluator verifica la presenza degli
assiomi e l'entailment della query dal solo insieme citato, ammettendo prove alternative.

Il **grounded score** è la quota di risposte originali corrette con supporto valido.
Il **withdrawal score** è la quota di ritiri corretti. Il **paired grounded revision score**
è la quota di coppie in cui entrambi i turni superano la verifica, sul totale dei casi selezionati.
Non si escludono dal denominatore le baseline sbagliate. Errori di formato contano
come insuccessi; errori infrastrutturali rendono il run non confrontabile.

Usare l'ambiente `requirements-belief-revision-full.txt` e il kernel `Python (epistemic pilot)`.
Modificare `N_QUERIES`, `SELECTION_SEED` e `MODELS` nel notebook. La cartella del dataset
viene scelta dalla configurazione; una cartella già presente viene verificata e riutilizzata,
mai sovrascritta. Le prime celle preparano il dataset; il ciclo su `MODELS` avvia le chiamate LLM.
**Run All** esegue anche i modelli: con 52 query sono 104 chiamate per modello.

Il builder verifica con HermiT tutto il pool prima di selezionare N query, senza ripetizioni,
con seed riproducibile. Non impone più quote per profondità o soggetti distinti. Un numero superiore
ai casi ammissibili genera un errore; nessun campione viene ridotto silenziosamente.
Sono registrati numero richiesto, casi ammissibili, esclusioni e distribuzione per profondità.
Gli score vengono riportati anche separatamente per profondità.

Per creare 25 casi e poi eseguirli (il percorso per `build` deve essere nuovo):

```powershell
.\.venv\Scripts\python.exe subclass_grounding_benchmark.py build --dataset benchmarks/subclass_grounding_v2_25 --num-queries 25
.\.venv\Scripts\python.exe subclass_grounding_benchmark.py run --dataset benchmarks/subclass_grounding_v2_25 --model qwen2.5-coder:7b
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_subclass_grounding.py -v
```

Per eseguire tutte le query già preparate:

```powershell
.\.venv\Scripts\python.exe subclass_grounding_benchmark.py run --dataset benchmarks/subclass_grounding_v2_all --model qwen2.5-coder:7b
```

`--num-queries all` è il default di `build`; `--seed` controlla il campionamento.
Queste opzioni non cambiano un run: ogni run usa integralmente il suo dataset congelato.
`run` senza `--dataset` continua a usare il pilot v1 per compatibilità.

I run sono salvati in `generated/subclass_grounding/<run_id>/`: `manifest.json`,
`responses.jsonl`, `evaluation.json` e copia degli snapshot. `compare_runs` rifiuta
confronti fra dataset, prompt, opzioni o evaluator diversi. Il builder rifiuta di
sovrascrivere un dataset esistente; per un nuovo dataset usare una directory/versione nuova.
L'aggiornamento del codice cambia il fingerprint: per confronti nuovi rieseguire i modelli con la stessa versione.
Questo pilot non misura ancora il comportamento su modifiche irrilevanti o supporti
alternativi sopravvissuti al ritiro, e non costituisce una graduatoria statistica.

## Valutare i risultati storici

Dalla radice del repository, con Python 3.11 o successivo (ambiente sorgente: Python 3.14.4):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate_balanced50.py
```

Su Windows attivare `.venv\Scripts\activate` al posto di `source`. Le metriche e la figura vengono scritte in `generated/evaluation/`, ignorata da Git. Questa operazione non richiede Ollama, Java o API remote. Le versioni delle dipendenze dirette sono quelle installate nell'ambiente sorgente; non è un lock completo delle dipendenze transitive e l'installazione su un ambiente pulito non è stata verificata.

## Ripetere l'esperimento con il modello

Installare e avviare [Ollama](https://ollama.com/download), poi:

```bash
ollama pull qwen3.5:4b
python run_balanced50.py --output generated/run_01.jsonl
python evaluate_balanced50.py --input generated/run_01.jsonl
```

Il servizio Ollama deve essere raggiungibile (default localhost:11434; configurabile con OLLAMA_HOST). Il modello viene scaricato separatamente e non è incluso nel repository. Il run mantiene prompt, temperature=0.6 e campione dell'esperimento originale. Non è garantita la riproducibilità bit per bit: versione Ollama, digest del modello e hardware del run storico non sono documentati. Registrare queste informazioni per nuovi confronti. `--model` permette un altro modello; scegliere sempre un nuovo file di output quando si cambia modello o configurazione.

Il comando riprende un output esistente saltando le domande già registrate, compresi i record con errori (comportamento originale): controllarli prima di considerare concluso un run. Per ripetere tutto utilizzare un nuovo nome di file. I percorsi relativi passati ai comandi sono risolti rispetto alla cartella del repository.

## Notebook e generazione della ground truth

```bash
python -m pip install -r requirements-notebooks.txt
python -m ipykernel install --user --name owl-reasoning --display-name "OWL reasoning"
jupyter lab
```

Per owlapy/HermiT occorre una JVM/JDK a 64 bit, per esempio OpenJDK 17, con JAVA_HOME configurato e `java -version` funzionante. I reasoner Java sono distribuiti tramite owlapy. Nei notebook scegliere il kernel appena installato e lavorare dalla radice del repository.

- `owlapy_inference.ipynb`: esplorazione TBox/ABox, generazione Q&A con HermiT, domain/range, esempi Pizza e proof chain. I file `owlapy_generated_qa_extended.json` e `owlapy_generated_qa_with_chains.json` sono output da generare, non erano presenti nella directory sorgente.
- `owlapy_queries_testing.ipynb`: notebook esplorativo completo con subclass Turtle/XML, domain/range e confronto degli assiomi. Le celle dipendono dallo stato del kernel e alcune lanciano inferenza o sovrascrivono dataset/risultati: non usare Run All per la sola valutazione. Per l'ultimo esperimento preferire i due script isolati.
- `subclass_justification.py`: giustificazioni mediante percorsi su subclass e componenti nominate delle intersezioni equivalenti.
- `axiom_match.py`: parsing e normalizzazione delle citazioni.

I notebook sono stati copiati senza output e metadata di esecuzione; un esempio opzionale commentato per un provider remoto è stato omesso. I due script sono estratti dalle celle 23–25 del notebook originale (indici zero-based), con argomenti CLI e output separati. Il protocollo scientifico è rimasto invariato.

## Dati inclusi

- `univ-bench-inst.ttl`, `.txt`, `.xml`, `_xml.txt`: ontologia/contesto nelle serializzazioni usate.
- `univ-bench.owl`, `University0_0.owl`, `data/pizza.ttl`: altre ontologie e dati ABox di supporto.
- `owlapy_generated_qa*.json`: dataset disponibili generati dal reasoner.
- `subclass_balanced_sample_50.json` e `.csv`: campione congelato (17 direct, 17 inferred, 16 negative). Bilanciato per tipo, non per True/False.
- `results_qwen_3_5_4b_subclass_balanced50_axioms_ttl.jsonl`: 50 risposte, tracce, assiomi citati, `gt_axioms`, `gt_steps`, risposta grezza e tempi. È il riferimento principale.
- Altri `results_qwen_3_5_4b*.jsonl`: run precedenti subclass Turtle (66 record), XML (152), domain/range (56). Numerosità diverse: non confrontare le percentuali senza allineare le domande.
- `axiom_eval_qwen_3_5_4b_subclass_balanced50.csv` e PNG: risultati di valutazione originali.
- `MANIFEST.json`: provenienza e hash SHA-256 delle copie.

## Interpretazione e limiti

Il risultato salvato sulle 50 domande ha accuracy 1.0; sui positivi precision assiomi circa 0.956, recall 0.971, F1 0.961. La metrica confronta una normalizzazione parziale delle citazioni, non l'equivalenza logica OWL completa. Le restrizioni anonime non sono verificate integralmente. Il costruttore di giustificazioni non è un reasoner OWL completo: assenza di percorso nel grafo ridotto non prova in generale non-entailment. Nei negativi il riferimento degli assiomi è vuoto; precision/exact match vanno letti come sovra-citazione, distintamente dalla groundedness. Le proof chain del notebook di generazione sono esplorative e non sostituiscono le giustificazioni del modulo dedicato.

## Collaborazione

Caricare questa cartella come radice del repository. Non sono inclusi ambiente virtuale, log JVM, credenziali, documenti amministrativi o vecchi filoni RAG/contraddizioni. Nessuna licenza nuova è stata assegnata al codice o ai dati: concordarla con gli autori e conservare gli avvisi presenti nelle ontologie LUBM/Pizza. Per nuovi run condividere configurazione, digest del modello e risultati selezionati in una cartella tracciata, senza sovrascrivere i riferimenti originali.

## Verifica del pacchetto

La valutazione offline è stata eseguita con successo nell’ambiente locale: il CSV rigenerato coincide con il riferimento, riga per riga e per tutti i campi. Verificati hash delle copie, sintassi Python, rimozione degli output notebook e principali pattern di credenziali. Non sono stati rieseguiti i run LLM né l’intero notebook HermiT; non è stata provata un’installazione pulita. JVM locale rilevata: OpenJDK 26.0.1.
