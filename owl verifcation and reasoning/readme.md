# owl verification and reasoning 
si verifica se quando si chiede a un llm di produrre una justification per la query cercano di cambiare il loro comportamento in modo che sia coerente. 

<p> si possono confrontare questi due esperimenti: </p>
<ul>
<li> Primo: si richiede inferenza su una query, senza richiedere spiegazione. Si registra punteggio di accuratezza sul dataset. 
<li> Secondo: su stesso dataset si richiede di produrre una <i>justification in owl da passare a un reasoner formale (es hermit)</i> per verificare se l'assioma oggetto della query è entailed or not.  
<li> <b> in entrambi i casi si analizza il thinking trace e si vede se segue lo stesso filone e stessi ragionamenti. Si calcola faithfulness e altre metriche judge based per natural language explanation. </b>


## per rendere i risultati statisticamente più robusti. 
TBD