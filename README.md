Le but de l’application est d’attribuer un score au candidat en tenant en compte toutes les informations présentes dans son cv, spécifiquement les technologies présentes dans son cv. 

Notre base de données Neo4j est alimenté par toutes les technologies inventées jusqu’à notre jour , cependant la longévité du produit nécessite de mettre à jour notre base de données constamment. 

Cependant,  si on suppose que 2 ans après le déploiement de l’application , un candidat présente dans son cv une technologie qui n’existe pas sur notre bdd. 

Nous serons dans l’obligation d’analyser cette technologie pour assurer une crédibilité totale en l’ajoutant à notre bdd.
Pour remédier à cette problématique, nous aurons besoin de passer par un workflow précis :
•	Vérifier l’existence de la technologie en cherchant son nom sur google.
•	Si la technologie existe nous devons normaliser le nom de la technologie en cherchant le format le plus utilisé dans la documentation.
•	Nous devons ensuite ajouter la technologie aux bdd redis et neo4j qui doivent être synchronisés
•	Finalement, nous devons créer les relations entre la technologie et les nœuds présents dans notre schéma Neo4j notamment , UTILIZES et USED_WITH

La solution proposée nécessite une approche qui dépasse une solution proprement algorithmique, nous allons donc recourir aux llms pour effectuer des tâches complexes. 

La solution comporte plusieurs étapes qui vont utilisé l’llm donc la solution qui se propose est de développer un worflow d’agents d’IA.
Nous avons enfin recourit à CREWAI.
