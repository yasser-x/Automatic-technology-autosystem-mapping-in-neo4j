from crewai import Task
from agents import TechVerificationAgent, TechNormalizationAgent, RelationshipVerificationAgent, create_graph_query_agent , GraphQueryAgent
def verify_technology_task(agent:TechVerificationAgent, technology):
    return Task(
        description=f"Verify if a '{technology}'exists by searching and analyzing web content",
        agent=agent,
        expected_output="String indicating if the technology was verified",
        function=agent.verify_technology,
        input=technology
    )

def normalize_technology_task(agent:TechNormalizationAgent,technology):
    return Task(
        description="Normalize technology names, add them to the graph, and associate them with relevant category and use cases",
        agent=agent,
        expected_output="Normalized technology name and the id of the technology node in the graph",
        function=agent.run,
        input=technology
    )

def query_graph_task(agent:GraphQueryAgent, technology):
    return Task(
        description="Query the Neo4j graph to find libraries and frameworks related to the technology and update relationships",
        agent=agent,
        expected_output="String with updated relationships for the technology",
        function=agent.update_technology_relationships
        
    )

def verify_relationship_task(agent: RelationshipVerificationAgent):
    return Task(
        description="Verify and score relationships",
        agent=agent,
        expected_output="String with verified relationships and confidence scores",
        function=agent.verify_tech_relationship
    )