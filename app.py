import re
import sys
import asyncio
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

DOSSIER = Path(__file__).parent
MODELE = "qwen2.5:7b"

CONSIGNES = """Tu es un assistant de cuisine. Tu réponds en français.
- Pour toute question sur une recette, utilise TOUJOURS l'outil chercher_recettes.
  Ne propose que des recettes trouvées par cet outil, et cite le fichier source.
- Vérifie que chaque recette respecte TOUS les critères demandés (temps, régime...) avant de la proposer.
- Si une recette utilise des unités américaines (cups, tbsp, oz), convertis-les avec convertir_unites.
- Pour gérer la liste de courses, utilise ajouter_courses et voir_courses.
- Si l'information n'est pas dans les recettes, dis-le honnêtement."""


# ---------- 1. Ressources chargées une seule fois ----------
@st.cache_resource
def charger_retriever():
    documents = [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in sorted((DOSSIER / "recettes").glob("*.txt"))
    ]
    morceaux = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(documents)
    base = Chroma.from_documents(
        morceaux, embedding=OllamaEmbeddings(model="nomic-embed-text"), collection_name="recettes_app"
    )
    return base.as_retriever(search_kwargs={"k": 3})


@st.cache_resource
def charger_outils_mcp():
    client = MultiServerMCPClient({
        "cuisine": {
            "command": sys.executable,
            "args": [str(DOSSIER / "serveur_mcp.py")],
            "transport": "stdio",
        }
    })
    return asyncio.run(client.get_tools())


retriever = charger_retriever()
outils_mcp = charger_outils_mcp()


@tool
def chercher_recettes(requete: str) -> str:
    """Cherche dans le livre de recettes celles qui correspondent à une demande
    (ingrédients disponibles, type de plat, régime, temps de préparation).
    Renvoie le texte des recettes trouvées avec le nom du fichier source."""
    docs = retriever.invoke(requete)
    return "\n\n".join(f"[Source : {d.metadata['source']}]\n{d.page_content}" for d in docs)


# ---------- 2. Appel de l'agent ----------
async def repondre(historique):
    agent = create_agent(
        model=ChatOllama(model=MODELE, temperature=0),
        tools=[chercher_recettes, *outils_mcp],
        system_prompt=CONSIGNES,
    )
    resultat = await agent.ainvoke({"messages": historique})
    nouveaux = resultat["messages"][len(historique):]

    sources, outils = set(), []
    for m in nouveaux:
        if m.type == "tool":
            outils.append(m.name)
            if m.name == "chercher_recettes":
                sources.update(re.findall(r"\[Source : (.+?)\]", m.content))
    return nouveaux[-1].content, sorted(sources), outils


# ---------- 3. Interface ----------
st.set_page_config(page_title="Assistant cuisine", page_icon="🍳")
st.title("🍳 Assistant cuisine")
st.caption(f"RAG + agent LangGraph + outils MCP · {MODELE} via Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("🛒 Liste de courses")
    fichier_courses = DOSSIER / "courses.json"
    if fichier_courses.exists():
        import json
        liste = json.loads(fichier_courses.read_text())
        st.markdown("\n".join(f"- {i}" for i in liste) if liste else "_Vide_")
        if st.button("Vider la liste"):
            fichier_courses.unlink()
            st.rerun()
    else:
        st.markdown("_Vide_")
    st.divider()
    if st.button("Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📚 Sources : " + ", ".join(msg["sources"]))
        if msg.get("outils"):
            st.caption("🔧 Outils : " + " → ".join(msg["outils"]))

if question := st.chat_input("Ex : un plat végétarien rapide ?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    historique = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit..."):
            reponse, sources, outils = asyncio.run(repondre(historique))
        st.markdown(reponse)
        if sources:
            st.caption("📚 Sources : " + ", ".join(sources))
        if outils:
            st.caption("🔧 Outils : " + " → ".join(outils))

    st.session_state.messages.append(
        {"role": "assistant", "content": reponse, "sources": sources, "outils": outils}
    )
    st.rerun()