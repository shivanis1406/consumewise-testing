import streamlit as st
from openai import OpenAI
import json, os
import requests, time
from data_extractor import extract_data, find_product, get_product
from nutrient_analyzer import analyze_nutrients
from rda import find_nutrition
from typing import Dict, Any

#Used the @st.cache_resource decorator on this function. 
#This Streamlit decorator ensures that the function is only executed once and its result (the OpenAI client) is cached. 
#Subsequent calls to this function will return the cached client, avoiding unnecessary recreation.

@st.cache_resource
def get_openai_client():
    #Enable debug mode for testing only
    return True, OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@st.cache_resource
def get_backend_urls():
    data_extractor_url = "https://data-extractor-67qj89pa0-sonikas-projects-9936eaad.vercel.app/"
    return data_extractor_url

debug_mode, client = get_openai_client()
data_extractor_url = get_backend_urls()

def extract_data_from_product_image(image_links, data_extractor_url):
    response = extract_data(image_links)
    return response

def get_product_data_from_db(product_name, data_extractor_url):
    response = get_product(product_name)
    return response

def get_product_list(product_name_by_user, data_extractor_url):
    response = find_product(product_name_by_user)
    return response


def rda_analysis(product_info_from_db_nutritionalInformation: Dict[str, Any], 
                product_info_from_db_servingSize: float) -> Dict[str, Any]:
    """
    Analyze nutritional information and return RDA analysis data in a structured format.
    
    Args:
        product_info_from_db_nutritionalInformation: Dictionary containing nutritional information
        product_info_from_db_servingSize: Serving size value
        
    Returns:
        Dictionary containing nutrition per serving and user serving size
    """
    nutrient_name_list = [
        'energy', 'protein', 'carbohydrates', 'addedSugars', 'dietaryFiber',
        'totalFat', 'saturatedFat', 'monounsaturatedFat', 'polyunsaturatedFat',
        'transFat', 'sodium'
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You will be given nutritional information of a food product. 
                                Return the data in the exact JSON format specified in the schema, 
                                with all required fields."""
                },
                {
                    "role": "user",
                    "content": f"Nutritional content of food product is {json.dumps(product_info_from_db_nutritionalInformation)}. "
                              f"Extract the values of the following nutrients: {', '.join(nutrient_name_list)}."
                }
            ],
        response_format={"type": "json_schema", "json_schema": {
            "name": "Nutritional_Info_Label_Reader",
            "schema": {
                "type": "object",
                "properties": {
                    "energy": {"type": "number"},
                    "protein": {"type": "number"},
                    "carbohydrates": {"type": "number"},
                    "addedSugars": {"type": "number"},
                    "dietaryFiber": {"type": "number"},
                    "totalFat": {"type": "number"},
                    "saturatedFat": {"type": "number"},
                    "monounsaturatedFat": {"type": "number"},
                    "polyunsaturatedFat": {"type": "number"},
                    "transFat": {"type": "number"},
                    "sodium": {"type": "number"},
                    "servingSize": {"type": "number"},
                },
                "required": nutrient_name_list + ["servingSize"],
                "additionalProperties": False
            },
            "strict": True
        }}
        )
        
        # Parse the JSON response
        nutrition_data = json.loads(response.choices[0].message.content)
        
        # Validate that all required fields are present
        missing_fields = [field for field in nutrient_name_list + ["servingSize"] 
                         if field not in nutrition_data]
        if missing_fields:
            print(f"Missing required fields in API response: {missing_fields}")
        
        # Validate that all values are numbers
        non_numeric_fields = [field for field, value in nutrition_data.items() 
                            if not isinstance(value, (int, float))]
        if non_numeric_fields:
            raise ValueError(f"Non-numeric values found in fields: {non_numeric_fields}")
        
        return {
            'nutritionPerServing': nutrition_data,
            'userServingSize': product_info_from_db_servingSize
        }
        
    except Exception as e:
        # Log the error and raise it for proper handling
        print(f"Error in RDA analysis: {str(e)}")
        raise


def find_product_nutrients(product_info_from_db):
    #GET Response: {'_id': '6714f0487a0e96d7aae2e839',
    #'brandName': 'Parle', 'claims': ['This product does not contain gold'],
    #'fssaiLicenseNumbers': [10013022002253],
    #'ingredients': [{'metadata': '', 'name': 'Refined Wheat Flour (Maida)', 'percent': '63%'}, {'metadata': '', 'name': 'Sugar', 'percent': ''}, {'metadata': '', 'name': 'Refined Palm Oil', 'percent': ''}, {'metadata': '(Glucose, Levulose)', 'name': 'Invert Sugar Syrup', 'percent': ''}, {'metadata': 'I', 'name': 'Sugar Citric Acid', 'percent': ''}, {'metadata': '', 'name': 'Milk Solids', 'percent': '1%'}, {'metadata': '', 'name': 'Iodised Salt', 'percent': ''}, {'metadata': '503(I), 500 (I)', 'name': 'Raising Agents', 'percent': ''}, {'metadata': '1101 (i)', 'name': 'Flour Treatment Agent', 'percent': ''}, {'metadata': 'Diacetyl Tartaric and Fatty Acid Esters of Glycerol (of Vegetable Origin)', 'name': 'Emulsifier', 'percent': ''}, {'metadata': 'Vanilla', 'name': 'Artificial Flavouring Substances', 'percent': ''}],
    
    #'nutritionalInformation': [{'name': 'Energy', 'unit': 'kcal', 'values': [{'base': 'per 100 g','value': 462}]},
    #{'name': 'Protein', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 6.7}]},
    #{'name': 'Carbohydrate', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 76.0}, {'base': 'of which sugars', 'value': 26.9}]},
    #{'name': 'Fat', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 14.6}, {'base': 'Saturated Fat', 'value': 6.8}, {'base': 'Trans Fat', 'value': 0}]},
    #{'name': 'Total Sugars', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 27.7}]},
    #{'name': 'Added Sugars', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 26.9}]},
    #{'name': 'Cholesterol', 'unit': 'mg', 'values': [{'base': 'per 100 g', 'value': 0}]},
    #{'name': 'Sodium', 'unit': 'mg', 'values': [{'base': 'per 100 g', 'value': 281}]}],
    
    #'packagingSize': {'quantity': 82, 'unit': 'g'},
    #'productName': 'Parle-G Gold Biscuits',
    #'servingSize': {'quantity': 18.8, 'unit': 'g'},
    #'servingsPerPack': 3.98,
    #'shelfLife': '7 months from packaging'}

    product_type = None
    calories = None
    sugar = None
    total_sugar = None
    added_sugar = None
    salt = None
    serving_size = None

    if product_info_from_db["servingSize"]["unit"] == "g":
        product_type = "solid"
    elif product_info_from_db["servingSize"]["unit"] == "ml":
        product_type = "liquid"
    serving_size = product_info_from_db["servingSize"]["quantity"]

    for item in product_info_from_db["nutritionalInformation"]:
        if 'energy' in item['name'].lower():
            calories = item['values'][0]['value']
        if 'total sugar' in item['name'].lower():
            total_sugar = item['values'][0]['value']
        if 'added sugar' in item['name'].lower():
            added_sugar = item['values'][0]['value']
        if 'sugar' in item['name'].lower() and 'added sugar' not in item['name'].lower() and 'total sugar' not in item['name'].lower():
            sugar = item['values'][0]['value']
        if 'salt' in item['name'].lower():
            if salt is None:
                salt = 0
            salt += item['values'][0]['value']

    if salt is None:
        salt = 0
        for item in product_info_from_db["nutritionalInformation"]:
            if 'sodium' in item['name'].lower():
                salt += item['values'][0]['value']

    #How to get Salt?
    if added_sugar is not None and added_sugar > 0 and sugar is None:
        sugar = added_sugar
    elif total_sugar is not None and total_sugar > 0 and added_sugar is None and sugar is None:
        sugar = total_sugar

    return product_type, calories, sugar, salt, serving_size
    
# Initialize assistants and vector stores
# Function to initialize vector stores and assistants
@st.cache_resource
def initialize_assistants_and_vector_stores():
    #Processing Level
    global client
    assistant1 = client.beta.assistants.create(
      name="Processing Level",
      instructions="You are an expert dietician. Use you knowledge base to answer questions about the processing level of food product.",
      model="gpt-4o",
      tools=[{"type": "file_search"}],
      temperature=0,
      top_p = 0.85
      )
    
    #Harmful Ingredients
    assistant2 = client.beta.assistants.create(
      name="Harmful Ingredients",
      instructions="You are an expert dietician. Use you knowledge base to answer questions about the ingredients in food product.",
      model="gpt-4o",
      tools=[{"type": "file_search"}],
      temperature=0,
      top_p = 0.85
      )
    
    #Harmful Ingredients
    assistant3 = client.beta.assistants.create(
      name="Misleading Claims",
      instructions="You are an expert dietician. Use you knowledge base to answer questions about the misleading claims about food product.",
      model="gpt-4o",
      tools=[{"type": "file_search"}],
      temperature=0,
      top_p = 0.85
      )
    
    # Create a vector store
    vector_store1 = client.beta.vector_stores.create(name="Processing Level Vec")
    
    # Ready the files for upload to OpenAI
    file_paths = ["Processing_Level.docx"]
    file_streams = [open(path, "rb") for path in file_paths]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch1 = client.beta.vector_stores.file_batches.upload_and_poll(
      vector_store_id=vector_store1.id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch1.status)
    print(file_batch1.file_counts)
    
    # Create a vector store
    vector_store2 = client.beta.vector_stores.create(name="Harmful Ingredients Vec")
    
    # Ready the files for upload to OpenAI
    file_paths = ["Ingredients.docx"]
    file_streams = [open(path, "rb") for path in file_paths]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch2 = client.beta.vector_stores.file_batches.upload_and_poll(
      vector_store_id=vector_store2.id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch2.status)
    print(file_batch2.file_counts)
    
    # Create a vector store
    vector_store3 = client.beta.vector_stores.create(name="Misleading Claims Vec")
    
    # Ready the files for upload to OpenAI
    file_paths = ["MisLeading_Claims.docx"]
    file_streams = [open(path, "rb") for path in file_paths]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch3 = client.beta.vector_stores.file_batches.upload_and_poll(
      vector_store_id=vector_store3.id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch3.status)
    print(file_batch3.file_counts)
    
    #Processing Level
    assistant1 = client.beta.assistants.update(
      assistant_id=assistant1.id,
      tool_resources={"file_search": {"vector_store_ids": [vector_store1.id]}},
    )
    
    #harmful Ingredients
    assistant2 = client.beta.assistants.update(
      assistant_id=assistant2.id,
      tool_resources={"file_search": {"vector_store_ids": [vector_store2.id]}},
    )
    
    #Misleading Claims
    assistant3 = client.beta.assistants.update(
      assistant_id=assistant3.id,
      tool_resources={"file_search": {"vector_store_ids": [vector_store3.id]}},
    )
    return assistant1, assistant2, assistant3

assistant1, assistant2, assistant3 = initialize_assistants_and_vector_stores()

def analyze_nutrition_icmr_rda(nutrient_analysis, nutrient_analysis_rda):
    global debug_mode, client
    system_prompt = """
Task: Analyze the nutritional content of the food item and compare it to the Recommended Daily Allowance (RDA) or threshold limits defined by ICMR. Provide practical, contextual insights based on the following nutrients:

Nutrient Breakdown and Analysis:
Calories:

Compare the calorie content to a well-balanced meal.
Calculate how many meals' worth of calories the product contains, providing context for balanced eating.
Sugar & Salt:

Convert the amounts of sugar and salt into teaspoons to help users easily understand their daily intake.
Explain whether the levels exceed the ICMR-defined limits and what that means for overall health.
Fat & Calories:

Analyze fat content, specifying whether it is high or low in relation to a balanced diet.
Offer insights on how the fat and calorie levels may impact the user’s overall diet, including potential risks or benefits.
Contextual Insights:
For each nutrient, explain how its levels (whether high or low) affect health and diet balance.
Provide actionable recommendations for the user, suggesting healthier alternatives or adjustments to consumption if necessary.
Tailor the advice to the user's lifestyle, such as recommending lower intake if sedentary or suggesting other dietary considerations based on the product's composition.

Output Structure:
For each nutrient (Calories, Sugar, Salt, Fat), specify if the levels exceed or are below the RDA or ICMR threshold.
Provide clear, concise comparisons (e.g., sugar exceeds the RDA by 20%, equivalent to X teaspoons).    
    """

    user_prompt = f"""
Nutrition Analysis :
{nutrient_analysis}
{nutrient_analysis_rda}
"""
    if debug_mode:
        print(f"\nuser_prompt : \n {user_prompt}")
        
    completion = client.chat.completions.create(
        model="gpt-4o",  # Make sure to use an appropriate model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return completion.choices[0].message.content
    
def analyze_processing_level(ingredients, assistant_id):
    global debug_mode, client
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Categorize food product that has following ingredients: " + ', '.join(ingredients) + " into Group A, Group B, or Group C based on the document. The output must only be the group category name (Group A, Group B, or Group C) alongwith the reason behind assigning that respective category to the product. If the group category cannot be determined, output 'NOT FOUND'.",
            }
        ]
    )
    
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
        include=["step_details.tool_calls[*].file_search.results[*].content"]
    )
    
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    #citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, "")
        #if file_citation := getattr(annotation, "file_citation", None):
        #    cited_file = client.files.retrieve(file_citation.file_id)
        #    citations.append(f"[{index}] {cited_file.filename}")

    if debug_mode:
        print(message_content.value)
    processing_level_str = message_content.value
    return processing_level_str

def analyze_harmful_ingredients(ingredients, assistant_id):
    global debug_mode, client
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "A food product has the following ingredients: " + ', '.join(ingredients) + ". Which are the harmful ingredients in this list? The output must be in JSON format: {<ingredient_name>: <information from the document about why ingredient is harmful>}. If information about an ingredient is not found in the documents, the value for that ingredient must start with the prefix '(NOT FOUND IN DOCUMENT)' followed by the LLM's response based on its own knowledge.",
            }
        ]
    )
    
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
        include=["step_details.tool_calls[*].file_search.results[*].content"]
    )
    
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    message_content = messages[0].content[0].text
    annotations = message_content.annotations

    #citations = []

    #print(f"Length of annotations is {len(annotations)}")

    for index, annotation in enumerate(annotations):
      if file_citation := getattr(annotation, "file_citation", None):
          #cited_file = client.files.retrieve(file_citation.file_id)
          #citations.append(f"[{index}] {cited_file.filename}")
          message_content.value = message_content.value.replace(annotation.text, "")
  
    if debug_mode:
      ingredients_not_found_in_doc = []
      print(message_content.value)
      for key, value in json.loads(message_content.value.replace("```", "").replace("json", "")).items():
          if value.startswith("(NOT FOUND IN DOCUMENT)"):
              ingredients_not_found_in_doc.append(key)
      print(f"Ingredients not found in the harmful ingredients doc are {','.join(ingredients_not_found_in_doc)}")
    harmful_ingredient_analysis = json.loads(message_content.value.replace("```", "").replace("json", "").replace("(NOT FOUND IN DOCUMENT) ", ""))
    
    harmful_ingredient_analysis_str = ""
    for key, value in harmful_ingredient_analysis.items():
      harmful_ingredient_analysis_str += f"{key}: {value}\n"
    return harmful_ingredient_analysis_str

def analyze_claims(claims, ingredients, assistant_id):
    global debug_mode, client
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "A food product named has the following claims: " + ', '.join(claims) + " and ingredients: " + ', '.join(ingredients) + """. Please evaluate the validity of each claim as well as assess if the product name is misleading.
The output must be in JSON format as follows: 

{
  <claim_name>: {
    'Verdict': <A judgment on the claim's accuracy, ranging from 'Accurate' to varying degrees of 'Misleading'>,
    'Why?': <A concise, bulleted summary explaining the specific ingredients or aspects contributing to the discrepancy>,
    'Detailed Analysis': <An in-depth explanation of the claim, incorporating relevant regulatory guidelines and health perspectives to support the verdict>
  }
}
"""
            }
                ]
    )
    
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
        include=["step_details.tool_calls[*].file_search.results[*].content"]
    )
    
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    
      
    annotations = message_content.annotations
    
    #citations = []
    
    #print(f"Length of annotations is {len(annotations)}")
    
    for index, annotation in enumerate(annotations):
          if file_citation := getattr(annotation, "file_citation", None):
              #cited_file = client.files.retrieve(file_citation.file_id)
              #citations.append(f"[{index}] {cited_file.filename}")
              message_content.value = message_content.value.replace(annotation.text, "")
      
    #if debug_mode:
    #    claims_not_found_in_doc = []
    #    print(message_content.value)
    #    for key, value in json.loads(message_content.value.replace("```", "").replace("json", "")).items():
    #          if value.startswith("(NOT FOUND IN DOCUMENT)"):
    #              claims_not_found_in_doc.append(key)
    #    print(f"Claims not found in the doc are {','.join(claims_not_found_in_doc)}")
    #claims_analysis = json.loads(message_content.value.replace("```", "").replace("json", "").replace("(NOT FOUND IN DOCUMENT) ", ""))
    claims_analysis = json.loads(message_content.value.replace("```", "").replace("json", ""))

    claims_analysis_str = ""
    for key, value in claims_analysis.items():
      claims_analysis_str += f"{key}: {value}\n"
    
    return claims_analysis_str

def generate_final_analysis(brand_name, product_name, nutritional_level, processing_level, harmful_ingredient_analysis, claims_analysis):
    global debug_mode, client
    system_prompt = """You are provided with a detailed analysis of a food product. Your task is to well-rounded analysis & generate actionable & practical recommendations for the user. 
Ensure the advice is personalised and practical. 

Use the following criteria to generate your response: 
1.⁠ ⁠Nutrition Analysis: 
--How much do sugar, calories, or salt exceed the ICMR threshold limit?
•⁠  ⁠How much of the Recommended Dietary Allowance (RDA) does the product provide for each nutrient?
•⁠  ⁠How processed is the product?
2.⁠ ⁠Harmful Ingredients: 
•⁠  ⁠Identify any harmful or questionable ingredients. 
3.⁠ ⁠Misleading Claims: 
•⁠  ⁠Are there any misleading claims made by the brand? 

After doing these analysis. before generating your actionable recommendation, form the whole consumption context of the product in the user's life.  
1. What is the consumption proportion of that product in the user's life - is it a whole meal or a snack, is it being cooked further possibly leading to addition of other nutrients, is it had at is? How much hunger does it satiate ? 

2. Discern the particular functionality of that product - is it the carb source in the meal, is it for caffeine, is it a travel essential, is it being had for protein? 

3. What is the perceived health benefit of that product?  Is the product being consumed for health reasons or as a treat? 

4. At what potential frequency would the user be having it? 

5. Could the consumer be overlooking hidden harms? Use the misleading claims criteria here and check if the consumer is having it thinking its healthier.  ⁠If the product is something they could consume daily, should they? ⁠If they are consuming it daily, what potential harm are they not noticing? ⁠If the product is intended for health purposes, are there concerns the user might miss? 


Final Recommendation Output:
Provide either of these potential practical recommendations, whatever seems most suited to the product's functionality & the analysis done above. 
1. Recommend whether the product is suitable for regular consumption or should be limited, considering healthier choices when applicable.
2. You also provide the recommendation considering other lifestyle habits - how well rounded are their meals, how active they are.
3. Recommend ideal frequency 
 4. Recommend functionality & meal type
5. Recommend how could they be balancing this with other meals or a particular kind of lifestyle 
6. How could they enhance this meal
7. You could also mention if a particular product could be seemed to be having for an instant sugar fix or a caffeine hit - that if you're doing it for this, once in a while it is okay. 
Be very practical yet empathetic. Think like an expert nutritionist and produce a well-rounded answer considering both good and bad and the consumption content. 

Keep the output very short, concise & factual. 

Follow this format: 
1.⁠ ⁠Recommendation: This will be the practical actionable recommendation  you provide  highlighting the most relevant point on which your analysis is based or the highest risk or benefit. 

2.⁠ ⁠Why - Here you provide the reasoning behind your take - the risks or benefits you noticed basis which you have given your recommendation. 
2.⁠ ⁠Risk Analysis: 
1.⁠ ⁠Nutrition: Do social math to contextualise the information. . Specify by how much do sugar, calories, or salt exceed the ICMR threshold limit? Contextualise the recommended daily allowance (RDA) information for the user-  provide information in teaspoons of sugar and salt or calories /fat equivalent to number of nutritious meals v/s the nutrition gained or the hunger satiated. Be very consistent with this.
2: Ingredients Analysis : Highlight the ingredients and mention the names of the specific ingredients on which your analysis was based and specify their harm or benefit. Also provide citations. 
3.⁠ ⁠Processing Level - Explain how processed is the food product 
4.⁠ ⁠Claims (if any): Is the claim valid or misleading giving detailed explanation


Be very short, concise and precise and scientific. Give actable insights & recommendations."""

    user_prompt = f"""
Product Name: {brand_name} {product_name}

Nutrition Analysis :
{nutritional_level}

Processing Level:
{processing_level}

Ingredient Analysis:
{harmful_ingredient_analysis}

Claims Analysis:
{claims_analysis}
"""
    if debug_mode:
        print(f"\nuser_prompt : \n {user_prompt}")
        
    completion = client.chat.completions.create(
        model="gpt-4o",  # Make sure to use an appropriate model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return f"{brand_name} {product_name} -- " + completion.choices[0].message.content


def analyze_product(product_info_raw):
    global assistant1, assistant2, assistant3
    
    if product_info_raw != "{}":
        product_info_from_db = json.loads(product_info_raw)
        brand_name = product_info_from_db.get("brandName", "")
        product_name = product_info_from_db.get("productName", "")
        ingredients_list = [ingredient["name"] for ingredient in product_info_from_db.get("ingredients", [])]
        claims_list = product_info_from_db.get("claims", [])
        nutritional_information = product_info_from_db['nutritionalInformation']
        serving_size = product_info_from_db["servingSize"]["quantity"]

        nutrient_analysis_rda = ""
        nutrient_analysis = ""
        nutritional_level = ""
        processing_level = ""
        harmful_ingredient_analysis = ""
        claims_analysis = ""
        
        if nutritional_information:
            product_type, calories, sugar, salt, serving_size = find_product_nutrients(product_info_from_db)
            nutrient_analysis = analyze_nutrients(product_type, calories, sugar, salt, serving_size)
            print(f"DEBUG ! nutrient analysis is {nutrient_analysis}")
            
            nutrient_analysis_rda_data = rda_analysis(nutritional_information, serving_size)
            print(f"DEBUG ! Data for RDA nutrient analysis is of type {type(nutrient_analysis_rda_data)} - {nutrient_analysis_rda_data}")
            print(f"DEBUG : nutrient_analysis_rda_data['nutritionPerServing'] : {nutrient_analysis_rda_data['nutritionPerServing']}")
            print(f"DEBUG : nutrient_analysis_rda_data['userServingSize'] : {nutrient_analysis_rda_data['userServingSize']}")

            if nutrient_analysis_rda_data:
                nutrient_analysis_rda = find_nutrition(nutrient_analysis_rda_data)
            print(f"DEBUG ! RDA nutrient analysis is {nutrient_analysis_rda}")
            #Call GPT for nutrient analysis
            nutritional_level = analyze_nutrition_icmr_rda(nutrient_analysis, nutrient_analysis_rda)
            
        if len(ingredients_list) > 0:    
            processing_level = analyze_processing_level(ingredients_list, assistant1.id) if ingredients_list else ""
            harmful_ingredient_analysis = analyze_harmful_ingredients(ingredients_list, assistant2.id) if ingredients_list else ""
            
        if len(claims_list) > 0:                    
            claims_analysis = analyze_claims(claims_list, ingredients_list, assistant3.id) if claims_list else ""
        
        final_analysis = generate_final_analysis(brand_name, product_name, nutritional_level, processing_level, harmful_ingredient_analysis, claims_analysis)
        if debug_mode:
            debug_information = f"""
            **Section 1**
            
Product Name: {brand_name} {product_name}

Ingredient List: 
{", ".join(ingredients_list)}

Salt/Sugar/Calories Exceed ICMR Limit?: 
{nutrient_analysis}

Nutrients Below RDA Limit?: 
{nutrient_analysis_rda}

Nutrient Analysis: 
{nutritional_level}

Processing Level: 
{processing_level}

Ingredient Analysis: 
{harmful_ingredient_analysis}

Claims Analysis: 
{claims_analysis}

            **Section 2**
"""
            return debug_information + final_analysis
        else:
            return final_analysis
    else:
        return "I'm sorry, product information could not be extracted from the url."    

# Streamlit app
# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

def chatbot_response(image_urls_str, product_name_by_user, data_extractor_url, extract_info = True):
    # Process the user input and generate a response
    processing_level = ""
    harmful_ingredient_analysis = ""
    claims_analysis = ""
    image_urls = []
    if product_name_by_user != "":
        similar_product_list_json = get_product_list(product_name_by_user, data_extractor_url)
        
        if similar_product_list_json and extract_info == False:
            with st.spinner("Fetching product information from our database... This may take a moment."):
                print(f"similar_product_list_json : {similar_product_list_json}")
                if 'error' not in similar_product_list_json.keys():
                    similar_product_list = similar_product_list_json['products']
                    return similar_product_list, "Product list found from our database"
                else:
                    return [], "Product list not found"
            
        elif extract_info == True:
            with st.spinner("Analyzing the product... This may take a moment."):
                product_info_raw = get_product_data_from_db(product_name_by_user, data_extractor_url)
                print(f"DEBUG product_info_raw from name: {product_info_raw}")
                if 'error' not in json.loads(product_info_raw).keys():
                    final_analysis = analyze_product(product_info_raw)
                    return [], final_analysis
                else:
                    return [], f"Product information could not be extracted from our database because of {json.loads(product_info_raw)['error']}"
                
        else:
            return [], "Product not found in our database."
                
    elif "http:/" in image_urls_str.lower() or "https:/" in image_urls_str.lower():
        # Extract image URL from user input
        if "," not in image_urls_str:
            image_urls.append(image_urls_str)
        else:
            for url in image_urls_str.split(","):
                if "http:/" in url.lower() or "https:/" in url.lower():
                    image_urls.append(url)

        with st.spinner("Analyzing the product... This may take a moment."):
            product_info_raw = extract_data_from_product_image(image_urls, data_extractor_url)
            print(f"DEBUG product_info_raw from image : {product_info_raw}")
            if 'error' not in json.loads(product_info_raw).keys():
                final_analysis = analyze_product(product_info_raw)
                return [], final_analysis
            else:
                return [], f"Product information could not be extracted from the image because of {json.loads(product_info_raw)['error']}"

            
    else:
        return [], "I'm here to analyze food products. Please provide an image URL (Example : http://example.com/image.jpg) or product name (Example : Harvest Gold Bread)"

class SessionState:
    """Handles all session state variables in a centralized way"""
    @staticmethod
    def initialize():
        initial_states = {
            "messages": [],
            "product_selected": False,
            "product_shared": False,
            "analyze_more": True,
            "welcome_shown": False,
            "yes_no_choice": None,
            "welcome_msg": "Welcome to ConsumeWise! What product would you like me to analyze today?",
            "similar_products": [],
            "awaiting_selection": False,
            "current_user_input": "",
            "selected_product": None
        }
        
        for key, value in initial_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

class ProductSelector:
    """Handles product selection logic"""
    @staticmethod
    def handle_selection():
        if st.session_state.similar_products:
            # Create a container for the selection UI
            selection_container = st.container()
            
            with selection_container:
                # Radio button for product selection
                choice = st.radio(
                    "Select a product:",
                    st.session_state.similar_products + ["None of the above"],
                    key="product_choice"
                )
                
                # Confirm button
                confirm_clicked = st.button("Confirm Selection")
                
                # Only process the selection when confirm is clicked
                if confirm_clicked:
                    if choice != "None of the above":
                        st.session_state.product_selected = True
                        st.session_state.awaiting_selection = False
                        st.session_state.selected_product = choice
                        _, msg = chatbot_response("", choice, data_extractor_url, extract_info=True)
                        st.session_state.messages.append({"role": "assistant", "content": msg})

                        # Loop through session state keys and delete all except the key_to_keep
                        keys_to_keep = ["messages", "welcome_msg"]
                        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
                    
                        for key in keys_to_delete:
                            del st.session_state[key]
                        st.session_state.welcome_msg = "What product would you like me to analyze next?"
                    else:
                        st.session_state.awaiting_selection = False
                        st.session_state.messages.append(
                            {"role": "assistant", "content": "Please provide the image URL of the product."}
                        )
                        
                    st.rerun()
                
                # Prevent further chat input while awaiting selection
                return True  # Indicates selection is in progress
            
        return False  # Indicates no selection in progress

class ChatManager:
    """Manages chat interactions and responses"""
    @staticmethod
    def process_response(user_input):
        if not st.session_state.product_selected:
            if "http:/" not in user_input and "https:/" not in user_input:
                return ChatManager._handle_product_name(user_input)
            else:
                return ChatManager._handle_product_url(user_input)
        return "Next Product"

    @staticmethod
    def _handle_product_name(user_input):
        st.session_state.product_shared = True
        st.session_state.current_user_input = user_input
        similar_products, _ = chatbot_response(
            "", user_input, data_extractor_url, extract_info=False
        )
        
        if similar_products:
            st.session_state.similar_products = similar_products
            st.session_state.awaiting_selection = True
            return "Here are some similar products from our database. Please select:"
        return "Product not found in our database. Please provide the image URL of the product."

    @staticmethod
    def _handle_product_url(user_input):
        is_valid_url = (".jpeg" in user_input or ".jpg" in user_input) and \
                       ("http:/" in user_input or "https:/" in user_input)
        
        if not st.session_state.product_shared:
            return "Please provide the product name first"
        
        if is_valid_url and st.session_state.product_shared:
            _, msg = chatbot_response(
                user_input, "", data_extractor_url, extract_info=True
            )
            st.session_state.product_selected = True
            return msg
            
        return "Please provide valid image URL of the product."

def main():
    # Initialize session state
    SessionState.initialize()
    
    # Display title
    st.title("ConsumeWise - Your Food Product Analysis Assistant")
    
    # Show welcome message
    if not st.session_state.welcome_shown:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": st.session_state.welcome_msg
        })
        st.session_state.welcome_shown = True
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle product selection if awaiting
    selection_in_progress = False
    if st.session_state.awaiting_selection:
        selection_in_progress = ProductSelector.handle_selection()
    
    # Only show chat input if not awaiting selection
    if not selection_in_progress:
        user_input = st.chat_input("Enter your message:", key="user_input")
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Process response
            response = ChatManager.process_response(user_input)
            
            if response == "Next Product":
                SessionState.initialize()  # Reset states for next product
                #st.session_state.welcome_msg = "What is the next product you would like me to analyze today?"
                keys_to_keep = ["messages", "welcome_msg"]
                keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
                    
                for key in keys_to_delete:
                    del st.session_state[key]
                st.session_state.welcome_msg = "What product would you like me to analyze next?"
                st.rerun()
                
            elif response:  # Only add response if it's not None
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                print(f"DEBUG : st.session_state.awaiting_selection : {st.session_state.awaiting_selection}")
                print(f"response : {response}")
                st.rerun()
    else:
        # Disable chat input while selection is in progress
        st.chat_input("Please confirm your selection above first...", disabled=True)
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    main()
