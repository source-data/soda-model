
from neo4j import GraphDatabase
import csv

# Connection parameters
uri = "bolt://127.0.0.1:7691"  # Adjust this to your Neo4j instance
username = "neo4j"
password = "pw_sourcedata"

# Initialize Neo4j connection
driver = GraphDatabase.driver(uri, auth=(username, password))

def fetch_data(query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        if not data:
            return
        fieldnames = data[0].keys()  # Assumes all records have the same keys
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in data:
            # Prepare the data for CSV output
            csv_record = {}
            for field in fieldnames:
                value = record[field]
                # Check if value is a Node, Relationship, or Path
                if isinstance(value, list):  # Handles lists, like a list of nodes
                    # Convert list of nodes or relationships to a string or other format
                    csv_record[field] = '; '.join([str(v.id) for v in value])  # Example: join IDs
                else:
                    csv_record[field] = value
            writer.writerow(csv_record)

# Define your queries
node_query = """
MATCH (articles:SDArticle)-->(connectedNodes)
WHERE connectedNodes: SDFigure OR connectedNodes: SDPanel OR connectedNodes: SDTag
WITH articles, collect(connectedNodes) AS nodes
LIMIT 100
RETURN articles, nodes
"""

# Fetch nodes
nodes = fetch_data(node_query)

# Write nodes to CSV
write_to_csv(nodes, "nodes.csv")

# Close the Neo4j driver
driver.close()
