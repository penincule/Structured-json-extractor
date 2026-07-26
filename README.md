# Extracteur d'informations LLM → sortie structurée

Un texte libre en entrée, du JSON propre et validé en sortie.

Stack : Python + [pydantic](https://docs.pydantic.dev/) + [instructor](https://github.com/instructor-ai/instructor) + l'API Claude (Anthropic). Un seul script, un seul appel API, un schéma.

## Exemple

Entrée (compte-rendu de match de rugby) :

> Dupont a marqué 2 essais, carton jaune à la 60e, Martin remplacé à la mi-temps...

Sortie :

```json
{
  "essais": [{ "joueur": "Dupont", "nombre": 2 }],
  "cartons": [{ "joueur": "Martin", "type": "jaune", "minute": 60 }],
  "remplacements": [{ "joueur_sortant": "Martin", "joueur_entrant": "...", "minute": 45 }]
}
```

Le schéma de sortie est forcé via un modèle Pydantic (`CompteRendu` dans `extract.py`), donc le JSON produit est toujours valide et typé.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # puis renseigner ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=sk-ant-...
```

## Utilisation

```bash
# Depuis un fichier
python extract.py exemple.txt

# Depuis stdin
cat exemple.txt | python extract.py

# Avec un autre modèle
python extract.py exemple.txt --model claude-sonnet-5
```

## Adapter à un autre domaine

Le schéma (`essais` / `cartons` / `remplacements`) est spécifique au rugby, à titre d'exemple. Pour l'adapter à une autre source de texte, il suffit de modifier les classes Pydantic dans `extract.py` — le reste du script (appel API, validation, sortie JSON) ne change pas.
