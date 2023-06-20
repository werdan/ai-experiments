import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
import pinecone
import numpy as np
import itertools
from tqdm import tqdm
from typing import List
from pydantic import BaseModel
import os
import json

# Define a Pydantic class to structure a model and convert it to JSON schema
class StepByStepAIResponse(BaseModel):  
    title: str  
    products_needed: List[str]

# Add OpenAI API key to environment
openai.api_key = os.getenv("OPENAI_API_KEY") 
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment='eu-west1-gcp')

# Create a Pinecone index
index = pinecone.Index("bauhaus-stg-products-paid")

# Read the CSV file into a DataFrame
projects_df = pd.read_csv('projects.csv')
projects_df = projects_df.head(1)

html_output = '''
<html>
    <head><title>DIY Projects created by AI with Bauhaus product references</title></head>
</html>
<body>
'''


# Iterate over the projects in the DataFrame
for _, project_row in projects_df.iterrows():
    project_name = project_row['Project Name']
    project_description = project_row['Description']

    # Call GPT-4 model to create product list needed of the project
    response = openai.ChatCompletion.create(  
        model="gpt-4-0613",  
        messages=[  
           {"role": "user", "content": project_description}  
        ],  
        functions=[  
            {  
              "name": "get_products_needed_for_the_project",  
              "description": "List products needed for the project, but limit products to five or less",  
              "parameters": StepByStepAIResponse.schema()  
            }  
        ],  
        function_call={"name": "get_products_needed_for_the_project"}  
    )

    # Parse the response
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])

    # Create a StepByStepAIResponse object from the response dict
    steps_obj = StepByStepAIResponse(**output)

    # For each step, fetch the closest products and append them to the step
    for i, step in enumerate(steps_obj.products_needed):
        # Create the embeddings with text-embedding-ada-002
        embeddings = get_embedding(step, engine='text-embedding-ada-002')

        # Convert the list of embeddings to a numpy array
        embeddings_array = np.array(embeddings).flatten()

        # Query the Pinecone index using the embeddings array
        query_result = index.query(vector=embeddings_array.tolist(), top_k=3, include_values=False, include_metadata=True)

        for result in query_result['matches']:
            product_id = result.get('id')
            metadata = result.get('metadata', {})

            # Check if metadata is present
            if metadata:
                product_name = metadata.get('name', 'N/A')
                product_description = metadata.get('description', 'N/A')
                product_image = metadata.get('base_image', 'N/A')
            else:
                product_name = product_description = product_image = 'N/A'

            # Append the suggested products to the step
            steps_obj.products_needed[i] += f"{product_name} URL: https://www.bauhaus.cz/{product_id}\nDescription: {product_description}\nImage: https://www.bauhaus.cz/img/150/150/resize/catalog/product/{product_image}\n"

    # Call GPT-4 model to re-create project description so that it makes sense with the products given
    prompt = '''
    Create a DIY project description based on the project name and product list.
    Format: 
     - Output format: HTML, but do not use <html> tag, as output will be embedded into a html template. 
     - For lists use <ul>.
     - Remove all \n symbols and replace with line breaks or preformatted text if needed. 

    Structure: 
     - Project name
     - Brief project description
     - Shopping list with product names, images and links to the online store and quantity when appropriate. Remove product references that do not match the step instructions. Never includes the product that is not related to the project context.
     - Describe each project step briefly.\n
'''
    project_with_products = prompt + "\n" + steps_obj.title + "\n" + "\n".join(steps_obj.products_needed)

    response = openai.ChatCompletion.create(  
        model="gpt-4-0613",  
        messages=[  
           {"role": "user", "content": project_with_products}  
        ]
    )

    html_output +=response.choices[0]['message']['content']
html_output +="</body></html>"
print(html_output)