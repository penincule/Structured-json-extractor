#!/usr/bin/env python3
"""Extracteur de texte libre vers JSON structuré, via l'API Claude.

Ce script fait aussi office de jeu d'évaluation : un jeu de comptes-rendus
annotés à la main (JEU_EVALUATION, ci-dessous) et un scorer qui compare la
sortie de l'extracteur à cette référence ("gold").

Utilisation :

  # Extraction (comportement par défaut)
  python extract.py exemple.txt
  cat exemple.txt | python extract.py

  # Score l'extracteur sur le jeu d'évaluation embarqué (appelle l'API)
  python extract.py --evaluer

  # Vérifie que le scorer lui-même est correct (prediction = gold, doit
  # donner 100 % partout — aucun appel API)
  python extract.py --oracle

Ce que mesure --evaluer :
- DETECTION  : precision / rappel / F1 par categorie — trouve-t-on les bons
               evenements sans en inventer ?
- ATTRIBUTS  : parmi les evenements bien detectes, les details (nombre,
               minute, entrant, type) sont-ils exacts ?
- CR exacts  : proportion de comptes-rendus entierement corrects.
"""

import argparse
import json
import os
import sys
import unicodedata
from typing import Literal

import anthropic
import instructor
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Schéma de sortie                                                            #
# --------------------------------------------------------------------------- #
class Essai(BaseModel):
    joueur: str = Field(
        description=(
            "Nom du joueur ayant marqué. Pour un essai de pénalité (aucun "
            "buteur nommé), utiliser 'essai de pénalité'."
        )
    )
    nombre: int = Field(description="Nombre d'essais marqués par ce joueur dans cet extrait")
    minute: int | None = Field(default=None, description="Minute de l'essai, si mentionnée")


class Carton(BaseModel):
    joueur: str
    type: Literal["jaune", "rouge"]
    minute: int | None = Field(default=None, description="Minute du carton, si mentionnée")


class Remplacement(BaseModel):
    joueur_sortant: str
    joueur_entrant: str | None = Field(
        default=None,
        description="Nom du joueur entrant, ou null si non mentionné (ex: sortie sur blessure)",
    )
    minute: int | None = Field(
        default=None,
        description="Minute du remplacement, si mentionnée. La mi-temps correspond à la minute 40.",
    )


class CompteRendu(BaseModel):
    essais: list[Essai] = Field(default_factory=list)
    cartons: list[Carton] = Field(default_factory=list)
    remplacements: list[Remplacement] = Field(default_factory=list)


def extract(texte: str, model: str = "claude-opus-5") -> CompteRendu:
    client = instructor.from_anthropic(anthropic.Anthropic())
    return client.chat.completions.create(
        model=model,
        max_tokens=2048,
        response_model=CompteRendu,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extrais les informations structurées de ce compte-rendu de match "
                    "de rugby. Conventions : la mi-temps correspond à la minute 40 ; "
                    "un essai de pénalité (sans buteur nommé) est un essai dont le "
                    "joueur est 'essai de pénalité' ; un joueur cité sans action "
                    "(essai, carton, remplacement) ne doit apparaître dans aucune "
                    "liste ; si aucun événement de ce type n'est mentionné, renvoyer "
                    "une liste vide plutôt que d'en inventer un.\n\n"
                    f"Compte-rendu :\n{texte}"
                ),
            }
        ],
    )


# --------------------------------------------------------------------------- #
# Jeu d'évaluation embarqué (14 comptes-rendus annotés)                       #
# --------------------------------------------------------------------------- #
JEU_EVALUATION = [
    {"id": "01_simple", "texte": "Dupont a marqué un essai en première mi-temps.",
     "attendu": {"essais": [{"joueur": "Dupont", "nombre": 1, "minute": None}], "cartons": [], "remplacements": []}},
    {"id": "02_minute", "texte": "Essai de Martin à la 23e minute, transformé dans la foulée.",
     "attendu": {"essais": [{"joueur": "Martin", "nombre": 1, "minute": 23}], "cartons": [], "remplacements": []}},
    {"id": "03_double", "texte": "Grosse performance de Lefebvre qui inscrit un doublé.",
     "attendu": {"essais": [{"joueur": "Lefebvre", "nombre": 2, "minute": None}], "cartons": [], "remplacements": []}},
    {"id": "04_mixte",
     "texte": "Dupont a marqué 2 essais, carton jaune à la 60e pour Bernard, et Martin remplacé à la mi-temps par Petit.",
     "attendu": {
         "essais": [{"joueur": "Dupont", "nombre": 2, "minute": None}],
         "cartons": [{"joueur": "Bernard", "type": "jaune", "minute": 60}],
         "remplacements": [{"joueur_sortant": "Martin", "joueur_entrant": "Petit", "minute": 40}],
     }},
    {"id": "05_jaune", "texte": "Carton jaune pour Moreau à la 34e pour un plaquage haut.",
     "attendu": {"essais": [], "cartons": [{"joueur": "Moreau", "type": "jaune", "minute": 34}], "remplacements": []}},
    {"id": "06_rouge", "texte": "Exclusion definitive de Garcia, carton rouge direct a la 71e minute.",
     "attendu": {"essais": [], "cartons": [{"joueur": "Garcia", "type": "rouge", "minute": 71}], "remplacements": []}},
    {"id": "07_remplacement", "texte": "A l'heure de jeu, Rousseau cede sa place a Fournier.",
     "attendu": {"essais": [], "cartons": [],
                 "remplacements": [{"joueur_sortant": "Rousseau", "joueur_entrant": "Fournier", "minute": 60}]}},
    {"id": "08_blessure",
     "texte": "Sortie sur blessure de Girard a la 52e, aucun remplacant annonce dans le compte-rendu.",
     "attendu": {"essais": [], "cartons": [],
                 "remplacements": [{"joueur_sortant": "Girard", "joueur_entrant": None, "minute": 52}]}},
    {"id": "09_vierge",
     "texte": "Match tres ferme et defensif, aucune realisation de part et d'autre. Score nul et vierge a la fin des 80 minutes.",
     "attendu": {"essais": [], "cartons": [], "remplacements": []}},
    {"id": "10_accents",
     "texte": "Essai de Gael a la 12e, puis Theo alourdit le score a la 48e. Muller ecope d'un carton jaune a la 65e.",
     "attendu": {
         "essais": [{"joueur": "Gael", "nombre": 1, "minute": 12}, {"joueur": "Theo", "nombre": 1, "minute": 48}],
         "cartons": [{"joueur": "Muller", "type": "jaune", "minute": 65}],
         "remplacements": [],
     }},
    {"id": "11_penalite",
     "texte": "Essai de penalite accorde par l'arbitre a la 78e apres une melee ecroulee volontairement.",
     "attendu": {"essais": [{"joueur": "essai de penalite", "nombre": 1, "minute": 78}], "cartons": [], "remplacements": []}},
    {"id": "12_multi_remp", "texte": "Double changement a la 55e : Blanc et Roux entrent a la place de Simon et Durand.",
     "attendu": {"essais": [], "cartons": [], "remplacements": [
         {"joueur_sortant": "Simon", "joueur_entrant": "Blanc", "minute": 55},
         {"joueur_sortant": "Durand", "joueur_entrant": "Roux", "minute": 55},
     ]}},
    {"id": "13_carton_et_essai", "texte": "Journee contrastee pour Leroy : un essai a la 15e puis un carton jaune a la 63e.",
     "attendu": {
         "essais": [{"joueur": "Leroy", "nombre": 1, "minute": 15}],
         "cartons": [{"joueur": "Leroy", "type": "jaune", "minute": 63}],
         "remplacements": [],
     }},
    {"id": "14_narratif",
     "texte": (
         "Le capitaine Faure a mene son equipe avec autorite mais n'a pas marque. "
         "C'est Chevalier qui a ouvert le score a la 8e, imite par Mercier a la 27e. "
         "Juste avant la pause, Robert a vu rouge apres un coup de coude (39e). "
         "En seconde periode, Faure a fait entrer Colin a la place de Perrin (58e), "
         "et Chevalier a signe son deuxieme essai a la 74e pour conclure."
     ),
     "attendu": {
         "essais": [{"joueur": "Chevalier", "nombre": 2, "minute": None}, {"joueur": "Mercier", "nombre": 1, "minute": 27}],
         "cartons": [{"joueur": "Robert", "type": "rouge", "minute": 39}],
         "remplacements": [{"joueur_sortant": "Perrin", "joueur_entrant": "Colin", "minute": 58}],
     }},
]


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #
def norm(s):
    """Minuscule, sans accents, espaces réduits — pour comparer des noms."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def to_dict(obj):
    """Accepte un modèle Pydantic ou un dict, renvoie un dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def apparier(golds, preds, key_fn):
    """
    Apparie les éléments de 'preds' aux éléments de 'golds' partageant la même
    clé. Renvoie (pairs, faux_positifs, faux_negatifs).
    """
    restant = list(golds)
    pairs, fp = [], []
    for p in preds:
        cible = None
        for g in restant:
            if key_fn(g) == key_fn(p):
                cible = g
                break
        if cible is not None:
            restant.remove(cible)
            pairs.append((cible, p))
        else:
            fp.append(p)
    return pairs, fp, restant  # restant = faux négatifs


CLES = {
    "essais": lambda e: norm(e.get("joueur")),
    "cartons": lambda c: (norm(c.get("joueur")), norm(c.get("type"))),
    "remplacements": lambda r: norm(r.get("joueur_sortant")),
}

# attributs à vérifier une fois l'événement correctement détecté
ATTRS = {
    "essais": ["nombre", "minute"],
    "cartons": ["minute"],
    "remplacements": ["joueur_entrant", "minute"],
}


def attrs_ok(gold, pred, categorie):
    for a in ATTRS[categorie]:
        gv, pv = gold.get(a), pred.get(a)
        if a == "joueur_entrant":
            gv, pv = norm(gv), norm(pv)
        if gv != pv:
            return False
    return True


def evaluer(cas, predire):
    cats = ["essais", "cartons", "remplacements"]
    stats = {c: {"tp": 0, "fp": 0, "fn": 0, "attr_ok": 0, "attr_tot": 0} for c in cats}
    cr_exacts = 0
    invalides = 0

    for c in cas:
        gold = c["attendu"]
        try:
            pred = to_dict(predire(c["texte"]))
        except Exception as e:  # extraction en échec / schéma invalide
            invalides += 1
            print(f"  ! {c['id']} : extraction en échec ({e})", file=sys.stderr)
            for cat in cats:
                stats[cat]["fn"] += len(gold.get(cat, []))
            continue

        cr_ok = True
        for cat in cats:
            g = gold.get(cat, []) or []
            p = pred.get(cat, []) or []
            pairs, fp, fn = apparier(g, p, CLES[cat])
            stats[cat]["tp"] += len(pairs)
            stats[cat]["fp"] += len(fp)
            stats[cat]["fn"] += len(fn)
            if fp or fn:
                cr_ok = False
            for gpair, ppair in pairs:
                stats[cat]["attr_tot"] += 1
                if attrs_ok(gpair, ppair, cat):
                    stats[cat]["attr_ok"] += 1
                else:
                    cr_ok = False
        if cr_ok:
            cr_exacts += 1

    return stats, cr_exacts, invalides, len(cas)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def afficher(stats, cr_exacts, invalides, total):
    print("\n" + "=" * 64)
    print("  RESULTATS DE L'EVALUATION")
    print("=" * 64)
    print(f"\n  {'Categorie':<16}{'Prec.':>8}{'Rappel':>8}{'F1':>8}{'Attributs':>12}")
    print("  " + "-" * 52)

    TP = FP = FN = 0
    for cat, s in stats.items():
        TP += s["tp"]; FP += s["fp"]; FN += s["fn"]
        p, r, f = prf(s["tp"], s["fp"], s["fn"])
        attr = s["attr_ok"] / s["attr_tot"] if s["attr_tot"] else 1.0
        print(f"  {cat:<16}{p:>8.0%}{r:>8.0%}{f:>8.0%}{attr:>10.0%} ({s['attr_ok']}/{s['attr_tot']})")

    p, r, f = prf(TP, FP, FN)
    print("  " + "-" * 52)
    print(f"  {'MICRO-MOYENNE':<16}{p:>8.0%}{r:>8.0%}{f:>8.0%}")

    print("\n" + "-" * 64)
    print(f"  Comptes-rendus entierement corrects : {cr_exacts}/{total}  ({cr_exacts/total:.0%})")
    if invalides:
        print(f"  Extractions en echec (schema invalide) : {invalides}/{total}")
    else:
        print(f"  Extractions valides : {total}/{total}  (100%)")
    print("=" * 64 + "\n")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "fichier",
        nargs="?",
        help="Fichier texte à extraire (défaut : lecture stdin). Ignoré avec --evaluer/--oracle.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EXTRACT_MODEL", "claude-opus-5"),
        help="Modèle Claude à utiliser (défaut: claude-opus-5)",
    )
    parser.add_argument(
        "--evaluer",
        action="store_true",
        help="Score l'extracteur sur le jeu d'évaluation embarqué (appelle l'API)",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Vérifie que le scorer est correct (prediction = gold, doit donner 100%%, aucun appel API)",
    )
    args = parser.parse_args()

    if args.oracle:
        predire = lambda texte: next(c["attendu"] for c in JEU_EVALUATION if c["texte"] == texte)
        stats, cr_exacts, invalides, total = evaluer(JEU_EVALUATION, predire)
        afficher(stats, cr_exacts, invalides, total)
        return

    if args.evaluer:
        predire = lambda texte: extract(texte, model=args.model)
        stats, cr_exacts, invalides, total = evaluer(JEU_EVALUATION, predire)
        afficher(stats, cr_exacts, invalides, total)
        return

    if args.fichier:
        with open(args.fichier, encoding="utf-8") as f:
            texte = f.read()
    else:
        texte = sys.stdin.read()

    resultat = extract(texte, model=args.model)
    print(resultat.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    main()
