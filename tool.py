from langchain.agents import Tool
from langchain.tools.python.tool import PythonREPLTool  
import numpy as np



desc = (
    "use this tool when you need to predict the probabilities given a series of feature values in a Generalized Additive Model."
    "It will return the prediction result and probability. Your final answer must begin with the results returned and your thoughts about it,"
    " plus a conclusion based on the combination of the results returned and the description of the data set. To use the tool you must input "
    "a list consisting of the value of each feature. The value must be provided in question. For example, the question is if a person's  WorkClass is Private"
    ", a person's Age is 27, fnlwgt is 160178, Education is Some-college, EducationNum is 10, MaritalStatus is Divorced, Occupation is Adm-clerical"
    ", what is the prediction? you need to provide this tool with [' Private', 27, 160178,' Some-college', 10, ' Divorced', ' Adm-clerical']."
    "Remember to put quotes around the word in the list. And the order of elements should be consistent with the order of features in the data you see. "
)
  
#预测工具的函数
def forecast(input):
    sample = eval(input)
    result = ebm.predict(sample)
    probability = ebm.predict_proba(sample)
    if isinstance(sample[0], list):
        ans = "The prediction results for each row from top to bottom are as follows:\n"
        for i, s in enumerate(sample) :
            probe = np.array(probability[i], dtype=float)
            probe = max(probe[0],probe[1]) 
            ans += f"    The prediction result of row {i} in the your input is {result[i]}, and the probability that the result comes to fruition is {probe}\n"
        return ans
    else:
        probability = np.array(probability, dtype=float)  
        probability = max(probability[0][0],probability[0][1]) 
        return f"The prediction result is {result[0]}，and the probability that the result comes to fruition is {probability}" 

forecast_tool = Tool(
    name='Forecast',
    func=forecast,
    description=desc
)

python_tool = PythonREPLTool()

python_tool.description = ("Use this tool when you need to execute python commands and plot. Input should be a valid python command."
" If you want to see the output of a value, you should print it out with `print(...)`.You can keep reusing this tool until you get the final answer."
)

def final(input):
    return input

Final_answer = Tool(
    name='Final_answer',
    func=final,
    description="Use this if you want to respond directly to the human. Input should be what you want to respond and it will return the same. After using this tool, you must leave the results returned by this tool intact as Final Answer instead of continuing with other actions."
)



"""class forecast_tool:
    name: str = 'Forecast'
    description: str = (
        "use this tool when you need to predict the probabilities given a series of feature values in a Generalized Additive Model."
        "It will return the prediction result and probability. Your final answer must begin with the results returned and your thoughts about it,"
        " plus a conclusion based on the combination of the results returned and the description of the data set. To use the tool you must input "
        "a list consisting of the value of each feature. The value must be provided in question. For example, the question is if a person's  WorkClass is Private"
        ", a person's Age is 27, fnlwgt is 160178, Education is Some-college, EducationNum is 10, MaritalStatus is Divorced, Occupation is Adm-clerical"
        ", what is the prediction? you need to provide this tool with [' Private', 27, 160178,' Some-college', 10, ' Divorced', ' Adm-clerical']."
        "Remember to put quotes around the word in the list. And the order of elements should be consistent with the order of features in the data you see. "
    ) 
"""