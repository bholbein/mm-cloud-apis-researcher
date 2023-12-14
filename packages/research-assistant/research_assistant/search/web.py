import json
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain.schema.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    ConfigurableField,
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.utilities import DuckDuckGoSearchAPIWrapper

RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()


def scrape_text(url: str):
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"


def web_search(query: str, num_results: int):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


get_links: Runnable[Any, Any] = (
    RunnablePassthrough()
    | RunnableLambda(
        lambda x: [
            {"url": url, "question": x["question"]}
            for url in web_search(query=x["question"], num_results=RESULTS_PER_QUESTION)
        ]
    )
).configurable_alternatives(
    ConfigurableField("search_engine"),
    default_key="duckduckgo",
    tavily=RunnableLambda(lambda x: x["question"])
    | RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "results": TavilySearchAPIRetriever(k=RESULTS_PER_QUESTION),
        }
    )
    | RunnableLambda(
        lambda x: [
            {"url": result.metadata["source"], "question": x["question"]}
            for result in x["results"]
        ]
    ),
)


SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{agent_prompt}"),
        (
            "user",
            "Schreibe 5 Google-Suchanfragen, um online eine objektive Meinung zu folgender Frage zu finden: {question}\n"
            "Du musst mit einer Liste von Strings im folgenden Format antworten: "
            '["Anfrage 1", "Anfrage 2", "Anfrage 3"].'
        ),
    ]
)

AUTO_AGENT_INSTRUCTIONS = """
Diese Aufgabe beinhaltet die Recherche zu einem gegebenen Thema, unabhängig von seiner Komplexität oder der Verfügbarkeit einer definitiven Antwort. Die Recherche wird von einem spezifischen Agenten durchgeführt, definiert durch seinen Typ und seine Rolle, wobei jeder Agent unterschiedliche Anweisungen benötigt.
Agent
Der Agent wird durch das Gebiet des Themas und den spezifischen Namen des Agenten bestimmt, der zur Erforschung des gegebenen Themas genutzt werden könnte. Agenten sind nach ihrem Fachgebiet kategorisiert, und jeder Agententyp ist einem entsprechenden Emoji zugeordnet.

examples:
task: "Sollte ich in Apple-Aktien investieren?"
response: 
{
    "agent": "💰 Finanz Agent",
    "agent_role_prompt: "Du bist ein erfahrener Finanzanalyse-KI-Assistent. Dein Hauptziel ist es, umfassende, scharfsinnige, unparteiische und methodisch strukturierte Finanzberichte auf Basis der bereitgestellten Daten und Trends zu erstellen."
}
task: "Könnte der Weiterverkauf von Sneakern profitabel werden?"
response: 
{ 
    "agent":  "📈 Business Analyst Agent",
    "agent_role_prompt": "Du bist ein erfahrener KI-Business-Analyst-Assistent. Dein Hauptziel ist es, umfassende, aufschlussreiche, unparteiische und systematisch strukturierte Geschäftsberichte auf Basis von bereitgestellten Geschäftsdaten, Markttrends und strategischen Analysen zu produzieren"
}
task: "Was sind die interessantesten Orte in Tel Aviv?"
response:
{
    "agent:  "🌍 Reise Agent",
    "agent_role_prompt": "Du bist ein weltgereister KI-Reiseführer-Assistent. Dein Hauptzweck ist es, fesselnde, aufschlussreiche, unvoreingenommene und gut strukturierte Reiseberichte über gegebene Orte zu verfassen, einschließlich Geschichte, Attraktionen und kulturellen Einblicken."
}
"""  # noqa: E501
CHOOSE_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=AUTO_AGENT_INSTRUCTIONS), ("user", "task: {task}")]
)

SUMMARY_TEMPLATE = """{text} 

-----------

Mit dem obigen Text, beantworte kurz die folgende Frage:

> {question}
 
-----------
Falls die Frage nicht mit dem Text beantwortet werden kann, fasse den Text kurz zusammen. Schließe alle faktischen Informationen, Zahlen, Statistiken usw. ein, falls verfügbar."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

scrape_and_summarize: Runnable[Any, Any] = (
    RunnableParallel(
        {
            "question": lambda x: x["question"],
            "text": lambda x: scrape_text(x["url"])[:10000],
            "url": lambda x: x["url"],
        }
    )
    | RunnableParallel(
        {
            "summary": SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0) | StrOutputParser(),
            "url": lambda x: x["url"],
        }
    )
    | RunnableLambda(lambda x: f"Source Url: {x['url']}\nSummary: {x['summary']}")
)

multi_search = get_links | scrape_and_summarize.map() | (lambda x: "\n".join(x))


def load_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


search_query = SEARCH_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106",
                                          temperature=0) | StrOutputParser() | load_json
choose_agent = (
    CHOOSE_AGENT_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106",
                                     temperature=0) | StrOutputParser() | load_json
)

get_search_queries = (
    RunnablePassthrough().assign(
        agent_prompt=RunnableParallel({"task": lambda x: x})
        | choose_agent
        | (lambda x: x.get("agent_role_prompt"))
    )
    | search_query
)


chain = (
    get_search_queries
    | (lambda x: [{"question": q} for q in x])
    | multi_search.map()
    | (lambda x: "\n\n".join(x))
)
