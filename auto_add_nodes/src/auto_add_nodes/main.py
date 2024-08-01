from crewai import Crew
from agents import (
    TechVerificationAgent,
    TechNormalizationAgent,
    RelationshipVerificationAgent,
    create_graph_query_agent,
    llm
)
from tasks import (
    verify_technology_task,
    normalize_technology_task,
    query_graph_task,
    verify_relationship_task
)

def run_tech_addition_pipeline(initial_technology):
    # Instantiate agents
    tech_verification_agent = TechVerificationAgent(llm)
    tech_normalization_agent = TechNormalizationAgent()
    graph_query_agent = create_graph_query_agent()
    relationship_verification_agent = RelationshipVerificationAgent()

    verify_task = verify_technology_task(tech_verification_agent, initial_technology)
    
    normalize_task = normalize_technology_task(tech_normalization_agent, initial_technology)
    normalize_task.context = [verify_task]  

    query_task = query_graph_task(graph_query_agent, initial_technology)
    query_task.context = [normalize_task]  # Set context as a list containing the previous task
    
    verify_relationship_task_instance = verify_relationship_task(relationship_verification_agent)
    verify_relationship_task_instance.context = [query_task]  # Set context as a list containing the previous task

    

    # Create the crew with the linked tasks
    crew = Crew(
        agents=[
            tech_verification_agent,
            tech_normalization_agent,
            graph_query_agent,
        ],
        tasks=[
            verify_task,
            normalize_task,
            query_task,
        ]
    )
    return crew.kickoff()

# Run the crew

if __name__ == "__main__":
    initial_technology = input("Enter a technology to test: ")
    result = run_tech_addition_pipeline(initial_technology)
    print("Result:", result)