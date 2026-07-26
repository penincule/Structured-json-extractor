# Extracteur d'informations LLM → sortie structurée

Un texte libre en entrée, du JSON propre et validé en sortie.

Stack : Python + [pydantic](https://docs.pydantic.dev/) + [instructor](https://github.com/instructor-ai/instructor) + l'API Claude (Anthropic). Un seul fichier (`extract.py`), un seul appel API, un schéma — et le jeu d'évaluation embarqué dedans.

## Exemple

Entrée (`exemple.txt`) :

> Belle rencontre ce week-end : Dupont a marqué 2 essais, l'un en première mi-temps
> et l'autre juste après la pause. Martin a été averti d'un carton jaune à la 60e minute
> pour un plaquage haut. En seconde période, Lefèvre a été remplacé par Girard à la 55e minute,
> et Martin lui-même a cédé sa place à Bernard à la 70e minute.

Sortie (`python extract.py exemple.txt`) :

```json
{
  "essais": [{ "joueur": "Dupont", "nombre": 2 }],
  "cartons": [{ "joueur": "Martin", "type": "jaune", "minute": 60 }],
  "remplacements": [
    { "joueur_sortant": "Lefèvre", "joueur_entrant": "Girard", "minute": 55 },
    { "joueur_sortant": "Martin", "joueur_entrant": "Bernard", "minute": 70 }
  ]
}
```

Le schéma de sortie est forcé via des modèles Pydantic (`Essai`, `Carton`, `Remplacement`, `CompteRendu` dans `extract.py`), donc le JSON produit est toujours valide et typé.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
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

## Évaluation

Le jeu d'évaluation (14 comptes-rendus annotés à la main) et le scorer sont embarqués directement dans `extract.py` — pas de fichier séparé. Les cas couvrent ce qui fait trébucher un extracteur : cas vierge (aucun événement), doublé, carton rouge vs jaune, remplacement sur blessure sans entrant connu, essai de pénalité, accents, distracteurs (joueurs cités sans action).

```bash
# Score l'extracteur en conditions réelles (appelle l'API)
python extract.py --evaluer

# Vérifie que le scorer lui-même est correct (prediction = gold, doit donner
# 100 % partout — aucun appel API)
python extract.py --oracle
```

Deux niveaux de mesure sont reportés :
- **Détection** (précision / rappel / F1 par catégorie) — l'extracteur trouve-t-il les bons événements sans en inventer ?
- **Attributs** — parmi les événements bien détectés, les détails (minute, nombre, type de carton) sont-ils exacts ?
- **Taux de comptes-rendus entièrement corrects** — proportion de CR parfaits de bout en bout.

Cette séparation détection / attributs est ce qui distingue la conformité au schéma (toujours garantie par `instructor`) de la justesse du contenu (ce que l'éval mesure réellement).

Pour ajouter des cas de test, complétez la liste `JEU_EVALUATION` en haut du script.

## Structure du projet

```
extract.py       # schéma, extraction, jeu d'évaluation et scorer — tout est ici
requirements.txt # anthropic, instructor, pydantic
exemple.txt      # texte d'exemple pour tester rapidement
.env.example     # variable d'environnement attendue (ANTHROPIC_API_KEY)
```
