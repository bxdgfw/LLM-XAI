from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent
from langchain.tools import Tool
from langchain.agents import Tool, AgentExecutor
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.tools.python.tool import PythonREPLTool
import markdown  
from IPython.display import Image, display  
import numpy as np
import pandas as pd
from llm2ebm import feature_importances_to_text
import t2ebm
from tool import Final_answer,python_tool
from graph_desc import llm2graph_desc

#用md语法表示的图的字符串生成图
def md2img(text):
    # 使用Markdown库将Markdown文本转换为HTML  
    html_output = markdown.markdown(text)  
      
    # 解析HTML中的图片标签，并显示图片  
    def process_image_tags(html):  
        from bs4 import BeautifulSoup  
        soup = BeautifulSoup(html, 'html.parser')  
          
        # 找到所有的图片标签  
        img_tags = soup.find_all('img')  
          
        # 遍历图片标签，显示图片  
        for img in img_tags:  
            url = img['src']  
            alt_text = img.get('alt', '')  
              
            # 使用IPython.display模块的Image类显示图片  
            display(Image(url=url, alt=alt_text))  
      
    # 调用函数解析图片标签并显示图片  
    process_image_tags(html_output)  

def ebm_agent(llm,ebm,df = None,dataset_description = None,y_axis_description = None):

    #获取需要的ebm的属性
    feature_importances = feature_importances_to_text(ebm) 
    graphs = []
    graph_descriptions = []
    for feature_index in range(len(ebm.feature_names_in_)):       #获取ebm中的所有graph
        graphs.append(t2ebm.graphs.extract_graph(ebm, feature_index))
    graphs = [t2ebm.graphs.graph_to_text(graph,max_tokens=1500) for graph in graphs]
    graphs = [graph.replace("{", "(").replace("}", ")") for graph in graphs]

    
    #prompt template
    prefix = """You are an expert statistician and data scientist.
            
    Your task is complete some tasks about a Generalized Additive Model (GAM). The model consists of different graphs that contain the effect of a specific input feature.
    You are bad at math such as predict the probability. So you must directly use the tool when asked about math problems. 
    
    You will be given:
        - The global feature importances of the different features in the model.
        - The graphs for the different features in the model. There is exactly one graph for each feature in the model.
    """
    
    if dataset_description is None or dataset_description == '':
        prefix += "\n\nThese inputs will be given to you by the user."
    else:
        prefix += "\n\nThe user will first provide a general description of what the dataset is about. Then you will be given the feature importance scores and the graph of each feature."
    
    prefix +="""\nGraphs will be presented as a JSON object with keys representing the x-axis and values representing the y-axis. For continuous features, the keys are intervals that represent ranges where the function predicts the same value. For categorical features, each key represents a possible value that the feature can take.\n"""
    
    if y_axis_description is not None and len(y_axis_description) > 0:
        prefix +=y_axis_description
    
    suffix = ""
    
    if dataset_description is not None and len(dataset_description) > 0:
        suffix += "Here is the general description of the data set:\n" + dataset_description + """\nThe description of dataset ends.\n"""
        
    suffix += "\nHere is the sequence of each feature and the global feature importance. Be sure not to provide these importances for the tool directly for prediction. The input list of the Forecast tool must include and only include the features included below:\n\n" + feature_importances 
    suffix += """
    \nThe information of each graph is stored in a list variable "ft_graph" in the order of features. If you need to get the graph and other information of a certain feature, first call the Python_REPL tool to obtain the contents of the variable. 
    For example, assuming that the feature sequence is: name, age, time. If you want to obtain the graph information of the age feature, you need to call the Python_REPL tool and enter print(ft_graph[1]).
    
    Information about graphs will be represented in the following format:
        - The name of the feature depicted in the graph
        - The type of the feature (continuous, categorical, or boolean)
        - Mean values
        - Lower bounds of confidence interval
        - Upper bounds of confidence interval
    """
    suffix+="""
    \nIf you need to use the Python_REPL tool, you must comply with the following regulations:
        If you get an error, debug your code and try again.
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        
    Complete the objective as best you can. You have access to the following tools:"""
    #    Your Final Answer must start with the entire executed code, and then print the image information in the form of a string in the format of '![image description](image address, which can be a local link)'.
    #    You must save the generated image through code before you show it. In other words,You can't use 'plt.show()' 
    prefix = prefix + suffix
    
    suffix_no_df = """The Action must contain only the name of one tool, no extra words needed to form a sentence.
    
    Begin!
    
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""
    
    suffix_with_df = """
    The Action must contain only the name of one tool, no extra words needed to form a sentence.
    If you find after Thought that you know the answer that can eventually be returned, you must immediately give the result in the form of Final Answer: the final answer to the original input question.
    Because the results of tools cannot be seen by users. So if the results need to be displayed to the user, please use the results of the observation as part of the final answer, rather than just telling the user the information they need are shown above.
    
    Additionally, when using the tool Python_REPL, you can execute code to work with a pandas dataframe. The name of the dataframe is `df`.
    If you want to get the data in df, remember to use the 'print()' function. For example, if you want to view the data with row index 3, you need to execute 'print(df.iloc[3])'
    
    This is the result of `print(df.head())`:
    {df_head}
    
    Begin!
    
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    #预测工具的描述
    desc = (
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
        description=desc
    )

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
    
    #添加agent的工具
    tools=[]
    tools.append(python_tool)
    tools.append(forecast_tool)
    tools.append(Final_answer)
    tools.append(explanation_tool)

    if df is not None:
        python_tool.python_repl.locals={"df": df,"ft_graph":graphs}
        input_variables=["input", "chat_history", "agent_scratchpad","df_head"]
        suffix = suffix_with_df
    else:
        python_tool.python_repl.locals={"ft_graph":graphs}
        input_variables=["input", "chat_history", "agent_scratchpad"]
        suffix = suffix_no_df
    
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    if 'df_head' in input_variables:
        prompt = prompt.partial(df_head=str(df.head().to_markdown()))

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain,tools=tools, verbose=True)
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory,handle_parsing_errors=True
    )
