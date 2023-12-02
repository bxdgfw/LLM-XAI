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
from prompt import suffix_no_df,suffix_with_df,get_prefix


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

def get_agent(llm,ebm,df = None,dataset_description = None,y_axis_description = None):
    
    #获取需要的ebm的属性
    feature_importances = feature_importances_to_text(ebm) 
    global_explanation = global_explanation = ebm.explain_global().data
    
    #获取prompt的prefix部分
    prefix = get_prefix(feature_importances,dataset_description,y_axis_description)

    #预测工具的函数
    forecast_desc = (
        "use this tool when you need to predict the probabilities given a series of feature values in a Generalized Additive Model."
        "It will return the prediction result and probability. Your final answer must begin with the results returned and your thoughts about it,"
        " plus a conclusion based on the combination of the results returned and the description of the data set. To use the tool you must input "
        "a list consisting of the value of each feature. The value must be provided in question. For example, the question is: if a person's  Name is Jobs"
        ", height is 175cm, weight is 65kg, what is the prediction? you need to provide this tool with [' Jobs', 175, 65]. Remember to put quotes around the strings in the list."
        "And there must be a space before the strings within the quotation marks. Also the order of elements should be consistent with the order of features in the data you see. "
    )    
    
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
        python_tool.python_repl.locals={"df": df,"ft_graph":global_explanation}
        input_variables=["input", "chat_history", "agent_scratchpad","df_head"]
        suffix = suffix_with_df
    else:
        python_tool.python_repl.locals={"ft_graph":global_explanation}
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
