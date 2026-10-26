import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("cuisine")
FICHIER = Path(__file__).parent / "courses.json"

def _lire() -> list[str]:
    return json.loads(FICHIER.read_text()) if FICHIER.exists() else []

@mcp.tool()
def ajouter_courses(ingredients: list[str]) -> str:
    """Ajoute un ou plusieurs ingrédients à la liste de courses."""
    liste = _lire()
    nouveaux = [i for i in ingredients if i not in liste]
    FICHIER.write_text(json.dumps(liste + nouveaux, ensure_ascii=False))
    return f"Ajouté : {', '.join(nouveaux) or 'rien de nouveau'}. Liste : {len(liste) + len(nouveaux)} articles."

@mcp.tool()
def voir_courses() -> str:
    """Affiche la liste de courses actuelle."""
    liste = _lire()
    return "\n".join(f"- {i}" for i in liste) if liste else "La liste de courses est vide."

@mcp.tool()
def convertir_unites(quantite: float, unite: str, ingredient: str) -> str:
    """Convertit une unité américaine (cups, tbsp, oz) en grammes ou millilitres.
    Le résultat dépend de l'ingrédient (farine, sucre, lait...)."""
    grammes_par_cup = {"farine": 125, "sucre": 200, "lait": 240, "beurre": 225, "eau": 240}
    unite = unite.lower().rstrip("s")
    if unite == "cup":
        base = grammes_par_cup.get(ingredient.lower(), 240)
        resultat = quantite * base
    elif unite == "tbsp":
        resultat = quantite * 15
    elif unite == "oz":
        resultat = quantite * 28.35
    else:
        return f"Unité inconnue : {unite}. Unités gérées : cups, tbsp, oz."
    sortie = "ml" if ingredient.lower() in ("lait", "eau") else "g"
    return f"{quantite} {unite} de {ingredient} ≈ {resultat:.0f} {sortie}"

if __name__ == "__main__":
    mcp.run()