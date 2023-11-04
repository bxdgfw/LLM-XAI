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
from tool import forecast_tool,Final_answer,python_tool
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
    graphs = [t2ebm.graphs.graph_to_text(graph) for graph in graphs]
    graph_descriptions = [llm2graph_desc(llm,ebm,idx,dataset_description=dataset_description,y_axis_description=y_axis_description) for idx in range(len(ebm.feature_names_in_)) ]
    graph_descriptions = "\n\n".join(
        [
            ebm.feature_names_in_[idx] + ": " + graph_description
            for idx, graph_description in enumerate(graph_descriptions)
        ]
    )

    #prompt template
    prefix = """You are an expert statistician and data scientist.
            
    Your task is complete some tasks about a Generalized Additive Model (GAM). The model consists of different graphs that contain the effect of a specific input feature.
    You are bad at math such as predict the probability. So you must directly use the tool when asked about math problem. 

    You will be given:
        - The global feature importances of the different features in the model.
        - Summaries of the graphs for the different features in the model. There is exactly one graph for each feature in the model.
    """
    if dataset_description is None or dataset_description == '':
        prefix += "\n\nThese inputs will be given to you by the user."
    else:
        prefix += "\n\nThe user will first provide a general description of what the dataset is about. Then you will be given the feature importance scores and the summaries of the individual features."

    suffix = ""

    if dataset_description is not None and len(dataset_description) > 0:
        suffix += "Here is the general description of the data set\n" + dataset_description
    
    if y_axis_description is not None and len(y_axis_description) > 0:
        suffix += "\n" + y_axis_description           
    
    suffix += "\nHere are the global feature importances. Be sure not to provide these importances for the tool directly for prediction. The input list of the Forecast tool must include and only include the features included below:\n\n" + feature_importances 
    suffix += "\nHere are the descriptions of the different graphs.\n\n"
    suffix += graph_descriptions
    suffix+="""
    If you need to use the Python_REPL tool, you must comply with the following regulations:
        If you get an error, debug your code and try again.
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        You have to use altair library to plot charts. Don't save it and show the chart using the function chart.display() instead of chart.show().
        You can also add some required interactivity to the chart.
        When you update or modify the chart, you must make modifications on the original chart, which means that the existing parts of the original chart cannot be changed.
        
    Complete the objective as best you can. You have access to the following tools:"""
    """
    You must save the generated image through code before you show it. In other words, You can't use 'plt.show()' .
    Your Final Answer must start with the entire executed code, and then print the image information in the form of a string in the format of '![image description](image address, which can be a local link)'.
    """
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
    
    #添加agent的工具
    tools=[]
    tools.append(python_tool)
    tools.append(forecast_tool)
    tools.append(Final_answer)

    if df is not None:
        python_tool.python_repl.locals={"df": df}
        input_variables=["input", "chat_history", "agent_scratchpad","df_head"]
        suffix = suffix_with_df
    else:
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
