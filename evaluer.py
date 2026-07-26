#!/usr/bin/env python3
"""
Evaluation de l'extracteur LLM a sortie structuree.

Mesure a quel point la sortie de l'extracteur correspond a des comptes-rendus
annotes a la main (le "gold" dans jeu_evaluation.jsonl).

Trois modes d'utilisation :

  1) Scorer directement ton extracteur
     Ton fichier extract.py doit exposer une fonction qui prend un texte
     et renvoie soit un modele Pydantic (avec .model_dump()), soit un dict,
     au format {essais: [...], cartons: [...], remplacements: [...]}.

       python evaluer.py --extracteur extract:extract

     (ici "extract:extract" = module extract.py, fonction extract)

  2) Scorer un fichier de predictions deja generees
     JSONL avec une ligne {"id": ..., "prediction": {...}} par cas.

       python evaluer.py --predictions mes_predictions.jsonl

  3) Verifier que le scorer lui-meme est correct (oracle : prediction = gold)
     Doit afficher 100 % partout. Sert a valider le scoring avant de brancher
     le LLM.

       python evaluer.py --oracle

Conventions de scoring
----------------------
- essais         : apparies par joueur (nom normalise). Attribut verifie : nombre, minute.
- cartons        : apparies par (joueur, type). Attribut verifie : minute.
- remplacements  : apparies par sortant. Attributs verifies : entrant, minute.

Deux niveaux de mesure sont reportes :
- DETECTION  : a-t-on trouve les bons evenements (bon joueur / bon type) ?
               -> precision / rappel / F1 par categorie + micro-moyenne.
- ATTRIBUTS  : parmi les evenements correctement detectes, les details
               (nombre, minute, entrant) sont-ils exacts ?
- Taux d'exactitude par compte-rendu : proportion de CR entierement corrects.
"""

import argparse
import json
import sys
import unicodedata
from importlib import import_module
from pathlib import Path

DATA = Path(__file__).parent / "jeu_evaluation.jsonl"


# --------------------------------------------------------------------------- #
# Normalisation                                                               #
# --------------------------------------------------------------------------- #
def norm(s):
    """Minuscule, sans accents, espaces reduits — pour comparer des noms."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def to_dict(obj):
    """Accepte un modele Pydantic ou un dict, renvoie un dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


# --------------------------------------------------------------------------- #
# Appariement gold <-> prediction pour une categorie                          #
# --------------------------------------------------------------------------- #
def apparier(golds, preds, key_fn):
    """
    Apparie les elements de 'preds' aux elements de 'golds' partageant la meme
    cle. Renvoie (pairs, faux_positifs, faux_negatifs).
    pairs = liste de (gold, pred).
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
    return pairs, fp, restant  # restant = faux negatifs


CLES = {
    "essais":        lambda e: norm(e.get("joueur")),
    "cartons":       lambda c: (norm(c.get("joueur")), norm(c.get("type"))),
    "remplacements": lambda r: norm(r.get("joueur_sortant")),
}

# attributs a verifier une fois l'evenement correctement detecte
ATTRS = {
    "essais":        ["nombre", "minute"],
    "cartons":       ["minute"],
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


# --------------------------------------------------------------------------- #
# Boucle d'evaluation                                                         #
# --------------------------------------------------------------------------- #
def evaluer(cas, predire):
    cats = ["essais", "cartons", "remplacements"]
    stats = {c: {"tp": 0, "fp": 0, "fn": 0, "attr_ok": 0, "attr_tot": 0} for c in cats}
    cr_exacts = 0
    invalides = 0

    for c in cas:
        gold = c["attendu"]
        try:
            pred = to_dict(predire(c["texte"]))
        except Exception as e:            # extraction en echec / schema invalide
            invalides += 1
            print(f"  ! {c['id']} : extraction en echec ({e})", file=sys.stderr)
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
    print(f"\n  {'Categorie':<16}{'Prec.':>8}{'Rappel':>8}{'F1':>8}"
          f"{'Attributs':>12}")
    print("  " + "-" * 52)

    TP = FP = FN = 0
    for cat, s in stats.items():
        TP += s["tp"]; FP += s["fp"]; FN += s["fn"]
        p, r, f = prf(s["tp"], s["fp"], s["fn"])
        attr = s["attr_ok"] / s["attr_tot"] if s["attr_tot"] else 1.0
        print(f"  {cat:<16}{p:>8.0%}{r:>8.0%}{f:>8.0%}"
              f"{attr:>10.0%} ({s['attr_ok']}/{s['attr_tot']})")

    p, r, f = prf(TP, FP, FN)
    print("  " + "-" * 52)
    print(f"  {'MICRO-MOYENNE':<16}{p:>8.0%}{r:>8.0%}{f:>8.0%}")

    print("\n" + "-" * 64)
    print(f"  Comptes-rendus entierement corrects : "
          f"{cr_exacts}/{total}  ({cr_exacts/total:.0%})")
    if invalides:
        print(f"  Extractions en echec (schema invalide) : {invalides}/{total}")
    else:
        print(f"  Extractions valides : {total}/{total}  (100%)")
    print("=" * 64 + "\n")


# --------------------------------------------------------------------------- #
# Fabrication de la fonction de prediction selon le mode                       #
# --------------------------------------------------------------------------- #
def charger_cas():
    return [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]


def predicteur_depuis_extracteur(spec):
    mod_name, _, fn_name = spec.partition(":")
    if not fn_name:
        fn_name = "extract"
    sys.path.insert(0, str(Path.cwd()))
    fn = getattr(import_module(mod_name), fn_name)
    return fn


def predicteur_depuis_fichier(chemin):
    lignes = [json.loads(l) for l in Path(chemin).read_text(encoding="utf-8").splitlines() if l.strip()]
    table = {d["id"]: d["prediction"] for d in lignes}
    cas_par_texte = {c["texte"]: c["id"] for c in charger_cas()}
    def predire(texte):
        return table[cas_par_texte[texte]]
    return predire


def main():
    ap = argparse.ArgumentParser(description="Evaluation de l'extracteur LLM.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--extracteur", metavar="module:fonction",
                   help="ex: extract:extract")
    g.add_argument("--predictions", metavar="fichier.jsonl",
                   help="predictions pre-generees")
    g.add_argument("--oracle", action="store_true",
                   help="verifie le scorer : prediction = gold (doit donner 100%%)")
    args = ap.parse_args()

    cas = charger_cas()

    if args.oracle:
        predire = lambda texte: next(c["attendu"] for c in cas if c["texte"] == texte)
    elif args.extracteur:
        predire = predicteur_depuis_extracteur(args.extracteur)
    else:
        predire = predicteur_depuis_fichier(args.predictions)

    stats, cr_exacts, invalides, total = evaluer(cas, predire)
    afficher(stats, cr_exacts, invalides, total)


if __name__ == "__main__":
    main()
