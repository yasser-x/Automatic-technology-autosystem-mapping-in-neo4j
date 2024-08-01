from langchain import OpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

OPENAI_API_KEY = "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"

graph = Neo4jGraph(
    url="neo4j+s://a56c4909.databases.neo4j.io",
    username="neo4j",
    password="pv4Fc667g__QSWQ62EXjk1ZDV_yxtRPFjfDiCSdPL_8"
)

chain = GraphCypherQAChain.from_llm(
    OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    graph=graph,
    verbose=True
)

def query_graph_qa(question):
    return chain.run(question)