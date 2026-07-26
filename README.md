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

## Évaluation

`jeu_evaluation.jsonl` contient 14 comptes-rendus annotés à la main (sortie attendue), couvrant les cas qui font trébucher un extracteur : cas vierge (aucun événement), doublé, carton rouge vs jaune, remplacement sur blessure sans entrant connu, essai de pénalité, accents, distracteurs (joueurs cités sans action).

`evaluer.py` (sans dépendance, stdlib pure) mesure la performance de l'extracteur en comparant sa sortie au gold :

```bash
# Score l'extracteur en conditions réelles (appelle l'API)
python evaluer.py --extracteur extract:extract

# Vérifie que le scorer lui-même est correct (doit donner 100 % partout)
python evaluer.py --oracle

# Score des prédictions déjà générées, sans rappeler l'API
python evaluer.py --predictions mes_predictions.jsonl
```

Deux niveaux de mesure sont reportés :
- **Détection** (précision / rappel / F1 par catégorie) — l'extracteur trouve-t-il les bons événements sans en inventer ?
- **Attributs** — parmi les événements bien détectés, les détails (minute, nombre, type de carton) sont-ils exacts ?
- **Taux de comptes-rendus entièrement corrects** — proportion de CR parfaits de bout en bout.

Cette séparation détection / attributs est ce qui distingue la conformité au schéma (toujours garantie par `instructor`) de la justesse du contenu (ce que l'éval mesure réellement).
