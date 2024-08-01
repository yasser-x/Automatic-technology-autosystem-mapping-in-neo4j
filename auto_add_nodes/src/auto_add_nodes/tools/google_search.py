from googleapiclient.discovery import build

API_KEY = "AIzaSyA1wGvk8SzHKM_kRw507fSTBlBsZqApB3A"
SEARCH_ENGINE_ID = "64ce969c078954e87"

def search_technology(technology):
    service = build("customsearch", "v1", developerKey=API_KEY)
    result = service.cse().list(q=technology, cx=SEARCH_ENGINE_ID, num=1).execute()
    
    if "items" in result and len(result["items"]) > 0:
        first_result = result["items"][0]
        return {
            "title": first_result.get("title", ""),
            "link": first_result.get("link", ""),
            "snippet": first_result.get("snippet", "")
        }
    else:
        return None

# Example usage
if __name__ == "__main__":
    tech = "Python"
    result = search_technology(tech)
    if result:
        print(f"First result for {tech}:")
        print(f"Title: {result['title']}")
        print(f"Link: {result['link']}")
        print(f"Snippet: {result['snippet']}")
    else:
        print(f"No results found for {tech}")