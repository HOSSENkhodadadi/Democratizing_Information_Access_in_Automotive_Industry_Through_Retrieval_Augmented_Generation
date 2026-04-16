from groq import Groq
import os
import json


DEFINED_TASK =  """You are an AI assistant that helps responding the queires asked by a clinet in a automation company. The queries can be regarding the makes,models, brands, ... of the
specific car/cars, you will recieve a query and you will recieve around thirty rows of data that may contain the required info to respond the question. Respond the query given the rows 
, if the given rows does not include the required data, generate the required context is not available, then mention the row/rows you used as correct context.
"""
META_ENDPOINT = ""
INFERENCE_CREDENTIAL = ""

class Generator:
    def __init__(self, credential = INFERENCE_CREDENTIAL, defined_task = DEFINED_TASK):
        if not credential:
            raise Exception("A key should be provided to invoke the endpoint")
        self.credential = credential
        self.defined_task = defined_task
        

    def prompt_meta(self, prompt: str) -> str:
        client = Groq(
    api_key=os.environ.get(INFERENCE_CREDENTIAL),
            )
        completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
        {
            "role": "user",
            "content":  prompt
        }
        ],
        temperature=1,
        # max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
        )
        response_text = ""
        for chunk in completion:
            # print(chunk.choices[0].delta.content or "", end="")
            response_text += chunk.choices[0].delta.content or ""
        
        # bot_response = f" {response_text}"

        # print("Response:", response.choices[0].message.content)
        # print("Model:", response.model)
        # print("Usage:")
        # print("	Prompt tokens:", response.usage.prompt_tokens)
        # print("	Total tokens:", response.usage.total_tokens)
        # print("	Completion tokens:", response.usage.completion_tokens)

        return response_text

# response = prompt_meta("The following is those cars mentioned earlier, extract a reasonable question from it that simulates a question that maybe asked by a user from them:")
# print(response)
