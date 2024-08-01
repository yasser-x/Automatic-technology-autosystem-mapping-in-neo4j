from crewai import Agent
from langchain.tools import Tool
from neo4j import GraphDatabase
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
import logging
from langchain_openai import ChatOpenAI
import openai
from pydantic import BaseModel, Field
from typing import List, Any , Dict
import uuid
from langchain.prompts import PromptTemplate
from googleapiclient.errors import HttpError



# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
# Configure LM Studio for Bloke/Mistral
llm = ChatOpenAI(
    openai_api_key="lm-studio",
    openai_api_base="http://localhost:1234/v1",
    model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    temperature=0.7
)

# Neo4j configuration
URI = "neo4j+s://a56c4909.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "pv4Fc667g__QSWQ62EXjk1ZDV_yxtRPFjfDiCSdPL_8"
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

class TechVerificationAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            name="Tech Verification Agent",
            role="Technology Verification Specialist",
            goal="Verify if a technology exists by searching and analyzing web content",
            backstory="I am an agent specialized in verifying the existence of technologies by searching the web and analyzing content from the first web link to verify the technology",
            llm=llm,
            tools=[
                Tool(
                    name="VerifyTechnology",
                    func=self.verify_technology,
                    description="Verify if a technology exists using web scraping from the first website rendered by Google custom search"
                )
            ]
        )
    API_KEY: str = 'AIzaSyA1wGvk8SzHKM_kRw507fSTBlBsZqApB3A'
    SEARCH_ENGINE_ID:str = "64ce969c078954e87"

    @staticmethod
    def search_technology(technology: str):
        try:
            service = build("customsearch", "v1", developerKey="AIzaSyA1wGvk8SzHKM_kRw507fSTBlBsZqApB3A")
            result = service.cse().list(q=technology, cx="64ce969c078954e87", num=1).execute()
            
            if "items" in result and len(result["items"]) > 0:
                first_result = result["items"][0]
                return {
                    "title": first_result.get("title", ""),
                    "link": first_result.get("link", ""),
                    "snippet": first_result.get("snippet", "")
                }
            else:
                return None
        except HttpError as e:
            print(f"An error occurred during search: {e}")
            return None

    def verify_technology(self, technology: str) -> str:
        try:
            search_result = self.search_technology(technology)
            if not search_result:
                return f"Not verified: {technology} (No search results found)"

            first_result_url = search_result['link']

            response = requests.get(first_result_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            paragraphs = soup.select('p:not(header p, nav p, footer p)')
            paragraphs = paragraphs[:5]  # Limit to first 5 paragraphs
            content = " ".join(paragraphs)

            prompt_template = PromptTemplate(
                input_variables=["technology", "content"],
                template="""
                Based on the following paragraphs from a website, determine if '{technology}' is a real technology, programming language, software, framework, or any other tech-related concept.

                Website content:
                {content}

                Answer with 'Yes' if it is a real technology, or 'No' if it is not. Provide a brief explanation for your decision.
                Do not suggest any other actions or searches. Use only the information provided.

                Answer:
                """
            )

            prompt = prompt_template.format(technology=technology, content=content)
            response = self.llm(prompt)

            # Parse the LLM response
            lines = response.strip().split('\n')
            is_tech = any(line.lower().startswith('yes') for line in lines)
            explanation = ' '.join(line for line in lines if not line.lower().startswith('yes') and not line.lower().startswith('no'))

            if is_tech:
                return f"Verified: {technology}. {explanation}"
            else:
                return f"Not verified: {technology}. {explanation}"

        except requests.RequestException as e:
            return f"Verification inconclusive: {technology} (Error scraping webpage: {str(e)})"
        except Exception as e:
            return f"Verification failed: {technology} (Unexpected error: {str(e)})"

    def run(self, technology: str) -> dict:
        result = self.verify_technology(technology)
        return {"output": result}

        


class TechNormalizationAgent(Agent):
    categories: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)
    def __init__(self):
        super().__init__(
            name="Tech Normalization Agent",
            role="Technology Normalization Expert",
            goal="Normalize technology names, add them to the graph, and associate them with relevant category and use cases",
            backstory="I am an agent specialized in standardizing technology names, adding them to the graph database, and associating them with relevant category and use cases.",
            llm=llm,
            tools=[
                
                Tool(
                    name="AddTechnologyToGraph",
                    func=self.add_technology_to_graph,
                    description="Add technology to the graph with associated category and use cases. Input should be a single string representing the technology name."
                )
            ]
        )
        self.categories = [
            "Programming Languages","Web Frameworks","Mobile App Frameworks","Desktop Application Frameworks","Game Development Engines","Database Management Systems","Object-Relational Mapping (ORM) Tools","Content Management Systems (CMS)",
            "Version Control Systems","Integrated Development Environments (IDEs)","Code Editors","Build Tools","Package Managers","Continuous Integration/Continuous Deployment (CI/CD) Tools","Testing Frameworks",
            "Automation Tools","Cloud Platforms","Containerization Technologies","Orchestration Tools","Serverless Platforms","Frontend Libraries/Frameworks","Backend Frameworks","Full-Stack Frameworks",
            "Application Programming Interfaces (APIs)","API Development Tools","Microservices Frameworks","Message Brokers","Caching Systems",
            "Search Engines","Big Data Processing Frameworks","Machine Learning Libraries","Artificial Intelligence Platforms","Internet of Things (IoT) Platforms","Blockchain Frameworks",
            "Virtual Reality (VR) Development Kits","Augmented Reality (AR) Development Kits","Data Visualization Libraries","Networking Libraries","Security Frameworks",
            "Authentication and Authorization Libraries","Cryptography Libraries","Logging Frameworks","Monitoring and Observability Tools",
            "Performance Profiling Tools","Code Analysis Tools","Documentation Generation Tools","Dependency Injection Frameworks","Task Queue Systems",
            "Workflow Engines","Business Process Management (BPM) Tools"
        ]
        self.use_cases = [
            "Web Application Development", "Mobile App Development", "Desktop Software Development",
            "Enterprise Resource Planning (ERP) Systems", "Customer Relationship Management (CRM) Systems",
            "E-commerce Platforms", "Content Management", "Social Media Platforms",
            "Messaging and Communication Systems", "Data Analytics and Business Intelligence",
            "Machine Learning and Artificial Intelligence", "Internet of Things (IoT) Applications",
            "Cloud Computing and Services", "Game Development", "Virtual Reality (VR) Applications",
            "Augmented Reality (AR) Applications", "Blockchain and Cryptocurrency Systems",
            "Financial Technology (FinTech) Solutions", "Healthcare Information Systems",
            "Educational Technology (EdTech) Platforms", "Cybersecurity and Network Security",
            "DevOps and IT Operations", "Geographic Information Systems (GIS)",
            "Scientific Computing and Simulation", "Computer-Aided Design (CAD) and Engineering",
            "Video and Audio Processing", "Natural Language Processing", "Computer Vision Applications",
            "Robotics and Automation", "Supply Chain Management", "Human Resources Management Systems",
            "Project Management Tools", "Collaborative Work Platforms", "Real-time Systems",
            "Embedded Systems", "Operating Systems Development", "Compiler and Interpreter Design",
            "Database Management and Big Data", "Search Engine Technology", "Recommendation Systems",
            "Authentication and Identity Management", "Payment Processing Systems",
            "Digital Marketing and Advertising Platforms", "Content Delivery Networks (CDN)",
            "Distributed Computing", "Parallel Processing", "Quantum Computing", "Bioinformatics",
            "Energy Management Systems", "Smart Home Technologies", "Automotive Software Systems",
            "Aerospace and Aviation Software", "Telecommunications Systems",
            "Environmental Monitoring and Management", "Logistics and Transportation Management",
            "Inventory Management Systems", "Point of Sale (POS) Systems", "Digital Publishing Platforms",
            "Music Production and Audio Engineering", "Video Streaming Services"
        ]
    def normalize_tech(self, technology: str) -> str:
        prompt = f"""
        Normalize the following technology name to it's commonly known format:
        {technology}

        Return only the normalized name without any additional explanation.
        """
        
        normalized_name = self.llm(prompt).strip()
        return normalized_name
    

    def add_technology_to_graph(self, technology: str) -> str:
        if not isinstance(technology, str):
            raise ValueError("Technology must be a string")
        
        # Normalize the technology name
        normalized_tech = self.normalize_tech(technology)
        # Determine relevant category and use cases
        relevant_category = self.get_relevant_category(normalized_tech)
        relevant_use_cases = self.get_relevant_use_cases(normalized_tech)
        
        with driver.session() as session:
            # Check if the technology already exists
            result = session.run(
                "MATCH (t:Technology {name: $name}) "
                "RETURN t",
                name=normalized_tech
            )
            existing_tech = result.single()
            
            if existing_tech:
                # Update category and use cases if the technology already exists
                session.run(
                    "MATCH (t:Technology {name: $name}) "
                    "SET t.category = $category, t.use_cases = $use_cases",
                    name=normalized_tech,
                    category=relevant_category,
                    use_cases=relevant_use_cases
                )
                message = f"Technology '{normalized_tech}' already exists in the graph. Updated category: {relevant_category}, and use cases: {', '.join(relevant_use_cases)}"
            else:
                # Create new technology node with category and use cases
                session.run(
                    "CREATE (t:Technology {id: $id,name: $name, category: $category, use_cases: $use_cases})",
                    name=normalized_tech,
                    category=relevant_category,
                    use_cases=relevant_use_cases
                )
                message = f"Successfully added new technology '{normalized_tech}' to graph with category: {relevant_category}, and use cases: {', '.join(relevant_use_cases)}"
            
            print(message)
            return message

    def get_relevant_category(self, technology: str) -> str:
        prompt = f"""
        Given the technology '{technology}', which ONE of the following categories is most relevant?
        List of categories: {', '.join(self.categories)}
        
        Please return only the name of the single most relevant category.
        """
        response = llm.predict(prompt)
        print(response.strip())
        return response.strip()

    def get_relevant_use_cases(self, technology: str) -> List[str]:
        prompt = f"""
        Given the technology '{technology}', which of the following use cases are most relevant?
        List of use cases: {', '.join(self.use_cases)}
        
        Please return only the names of the relevant use cases, separated by commas.
        """
        response = llm.predict(prompt)
        relevant_use_cases = [case.strip() for case in response.split(',')]
        print([case for case in relevant_use_cases if case in self.use_cases])
        return [case for case in relevant_use_cases if case in self.use_cases]
    @staticmethod
    def run(self, technology: str) -> Dict[str, Any]:
        # This method is called by the CrewAI framework
        result = self.add_technology_to_graph(technology)
        return {"output": result}
class GraphQueryAgent(Agent, BaseModel):
    name: str = Field(default="Graph Query Agent")
    role: str = Field(default="Graph Database Query Specialist")
    goal: str = Field(default="Query the Neo4j graph to find libraries and frameworks related to the technology and update relationships")
    backstory: str = Field(default="I am an agent specialized in managing technology relationship data in graph databases.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        
        logger.debug("Initializing Neo4j graph connection")
        try:
            neo4j_url = os.getenv("NEO4J_URL", URI)
            neo4j_username = os.getenv("NEO4J_USERNAME", USERNAME)
            neo4j_password = os.getenv("NEO4J_PASSWORD", PASSWORD)
            
            logger.debug(f"Attempting to connect to Neo4j: URL={neo4j_url}, Username={neo4j_username}")
            
            self._graph = Neo4jGraph(
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password
            )
            logger.debug("Neo4j graph connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j graph connection: {e}")
            raise

        self._chain = GraphCypherQAChain.from_llm(
            llm,
            graph=self._graph,
            verbose=True
        )

        self.tools = [
            Tool(
                name="Update Technology Relationships",
                func=self.update_technology_relationships,
                description="Update relationships between technologies in the Neo4j graph"
            )
        ]

    def query_graph_qa(self, question):
        return self._chain.run(question)

    def get_all_technologies(self):
        query = """
        MATCH (t:Technology)
        RETURN t.name AS name
        """
        result = self.query_graph_qa(query)
        return [tech.strip() for tech in result.split('\n') if tech.strip()]

    def analyze_relationships(self, technology, all_technologies):
        prompt = f"""
        Given the technology '{technology}' and the following list of all technologies:
        {', '.join(all_technologies)}

        Determine which technologies could be frameworks or libraries for '{technology}', 
        or if '{technology}' could be a framework or library for any of them.

        Return the results in the following format:
        related_technology1:relationship_type
        related_technology2:relationship_type
        ...

        Where relationship_type is either 'is_framework_for' or 'is_library_for'.
        Only include relationships that you are confident about.
        """
        return self.query_graph_qa(prompt)

    def update_relationship(self, source, target, relationship_type):
        query = f"""
        MATCH (s:Technology {{name: '{source}'}}), (t:Technology {{name: '{target}'}})
        MERGE (s)-[r:USED_WITH]->(t)
        SET r.type = '{relationship_type}'
        """
        self.query_graph_qa(query)

    def update_technology_relationships(self, technology):
        if not isinstance(technology, str):
            raise ValueError("Technology must be a string.")
        all_technologies = self.get_all_technologies()
        relationships = self.analyze_relationships(technology, all_technologies)
        
        updated_relations = []
        for line in relationships.split('\n'):
            if ':' in line:
                related_tech, rel_type = line.split(':')
                related_tech = related_tech.strip()
                rel_type = rel_type.strip()
                
                if rel_type == 'is_framework_for' or rel_type == 'is_library_for':
                    self.update_relationship(technology, related_tech, rel_type)
                    updated_relations.append((related_tech, rel_type))
                else:
                    self.update_relationship(related_tech, technology, rel_type)
                    updated_relations.append((related_tech, f"reverse_{rel_type}"))

        result_str = ', '.join([f"{tech} ({rel_type})" for tech, rel_type in updated_relations])
        return f"Updated relationships for {technology}: {result_str}"
def create_graph_query_agent():
    return GraphQueryAgent()

class RelationshipVerificationAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Relationship Verification Agent",
            role='i m a relationship verification specialist',
            goal="Verify 'is_library' or 'is_framework' relationships between technologies and assign confidence scores",
            llm=llm,
            backstory="I am an agent specialized in verifying and scoring relationships in graph databases, focusing on library and framework relationships."
        )

    def verify_tech_relationship(self, relationships):
        verified_relationships = []
        for relationship in relationships:
            source, rel_type, target = relationship.split(" ")
            # Use OpenAI to generate a confidence score
            prompt = f"On a scale of 1 to 10, how confident are you that {source} {rel_type} {target}? Respond with just the number."
            response = llm.predict(prompt)
            confidence_score = int(response.choices[0].text.strip())
            
            with driver.session() as session:
                session.run(
                    f"MATCH (s:Technology {{name: $source}})-[r:{rel_type}]->(t:Technology {{name: $target}}) "
                    f"SET r.confidence_score = $score",
                    source=source, target=target, score=confidence_score
                )
            verified_relationships.append(f"{source} {rel_type} {target} (Confidence: {confidence_score})")
        return f"Verified relationships: {'; '.join(verified_relationships)}"