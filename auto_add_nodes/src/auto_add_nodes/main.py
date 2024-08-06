from crewai import Crew
from crewai import Task
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
    def normalize_with_context(agent, task_input):
        verification_result = task_input['context'][0].output
        return agent.run({
            'technology': initial_technology,
            'scraped_content': verification_result.get('content', '')
        })

    normalize_task = normalize_technology_task(tech_normalization_agent, initial_technology)
    normalize_task.context = [verify_task]
    normalize_task._output = normalize_with_context

    def query_with_context(agent, task_input):
        verification_result = task_input['context'][1].output
        normalization_result = task_input['context'][0].output
        return agent.update_technology_relationships({
            'technology': normalization_result.get('normalized_tech', initial_technology),
            'scraped_content': verification_result.get('content', '')
        })

    query_task = query_graph_task(graph_query_agent, initial_technology)
    query_task.context = [normalize_task, verify_task]
    query_task._output = query_with_context

    

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
    result = run_tech_addition_pipeline('Tensorflow')
