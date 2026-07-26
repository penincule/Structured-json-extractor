#!/usr/bin/env python3
"""Extracteur de texte libre vers JSON structuré, via l'API Claude."""

import argparse
import json
import os
import sys
from typing import Literal

import anthropic
import instructor
from pydantic import BaseModel, Field


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fichier",
        nargs="?",
        help="Fichier texte à traiter (par défaut : lecture depuis stdin)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EXTRACT_MODEL", "claude-opus-5"),
        help="Modèle Claude à utiliser (défaut: claude-opus-5)",
    )
    args = parser.parse_args()

    if args.fichier:
        with open(args.fichier, encoding="utf-8") as f:
            texte = f.read()
    else:
        texte = sys.stdin.read()

    resultat = extract(texte, model=args.model)
    print(resultat.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    main()
