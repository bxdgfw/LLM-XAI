from langchain.agents import Tool
from langchain.tools.python.tool import PythonREPLTool  
import numpy as np



forecast_desc = (
    "use this tool when you need to predict the probabilities given a series of feature values in a Generalized Additive Model."
    "It will return the prediction result and probability. Your final answer must begin with the results returned and your thoughts about it,"
    " plus a conclusion based on the combination of the results returned and the description of the data set. To use the tool you must input "
    "a list consisting of the value of each feature. The value must be provided in question. For example, the question is: if a person's  Name is Jobs"
    ", height is 175cm, weight is 65kg, what is the prediction? you need to provide this tool with [' Jobs', 175, 65]. Remember to put quotes around the strings in the list."
    "And there must be a space before the strings within the quotation marks. Also the order of elements should be consistent with the order of features in the data you see. "
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
    description=forecast_desc
)

python_tool = PythonREPLTool()

python_tool_desc = (
    "Use this tool when you need to execute python commands to obtain data or plot charts. Input should be a valid python command."
    " If you want to see the output of a value, you should print it out with `print(...)`.You can keep reusing this tool until you get the final answer."
    "For chart plotting, the altair library is mandatory. Instead of saving your chart, display it using the `chart.display()` function rather than `chart.show()`."
    "You can also add some required interactivity to the chart. When you update or modify the chart, you must make modifications on the original chart,"
    " which means that the existing parts of the original chart cannot be changed."
)

python_tool.description = python_tool_desc

#得到local explain的图工具
desc = (
    "use this tool When you need to get the contribution of each feature to the prediction outcome of a sample."
    "A positive score indicates a positive contribution to the prediction result, otherwise it is a negative contribution. "
    "The larger the absolute value of the score, the greater the degree of contribution."
)

def local_exp(input):
    sample = eval(input)
    explanation = ebm.explain_local(sample)
    if isinstance(sample[0], np.ndarray) or isinstance(sample[0], list):
        ans = f"The feature contribution to prediction results of features in each sample from top to bottom are as follows(The order of features corresponding to these scores is:{explanation.data(0)['names']}):\n"
        for i, s in enumerate(sample) :
            scores = [f'{x:.4f}' for x in explanation.data(i)['scores']]  
            ans += f"     the contribution scores of row {i} in your input is {scores}\n"
        return ans
    else:
        explanation = ebm.explain_local(sample)
        scores = [f'{x:.4f}' for x in explanation.data(0)['scores']] 
        return f"the contribution scores is {scores}, and the order of features corresponding to these scores is:{explanation.data(0)['names']}" 
        
explanation_tool = Tool(
    name='Explanation_tool',
    func=local_exp,
    description=desc
)

def final(input):
    return input

Final_answer = Tool(
    name='Final_answer',
    func=final,
    description="Use this if you want to respond directly to the human. Input should be what you want to respond and it will return the same. After using this tool, you must leave the results returned by this tool intact as Final Answer instead of continuing with other actions."
)
