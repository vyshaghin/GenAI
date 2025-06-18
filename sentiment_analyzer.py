from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langfuse import Langfuse
import yfinance as yf
import os
import json
from langfuse.langchain import CallbackHandler
# Set up environment variables for Langfuse
os.environ["LANGFUSE_PUBLIC_KEY"] = "" 
os.environ["LANGFUSE_SECRET_KEY"] = "" 
os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"

 
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

# Step 1: Accept company input
dynamic_input = input("Enter a company name (e.g., Microsoft): ")
company_name = dynamic_input.strip()

try:
    search_results = yf.Ticker(company_name)
    stock_code = search_results.info.get("symbol")
    if not stock_code:
        raise ValueError("Stock symbol not found.")
    else:
        news_tool = YahooFinanceNewsTool()
        news_raw = news_tool.run(stock_code)
        news_summary = news_raw[:1000]  # Truncate if needed
except Exception as e:
    raise ValueError(f"Error fetching stock symbol: {e}")


llm = AzureChatOpenAI(
    deployment_name="myllm",
    temperature=0.2,
    openai_api_key="601031146c4745a88900a80175a1aa51",
    openai_api_base="https://eastus.api.cognitive.microsoft.com/",
    openai_api_version="2024-12-01-preview"
)
response_schemas = [
    ResponseSchema(name="company_name", description="The input company name."),
    ResponseSchema(name="stock_code", description="The stock symbol."),
    ResponseSchema(name="newsdesc", description="Summary of recent news."),
    ResponseSchema(name="sentiment", description="Positive/Negative/Neutral"),
    ResponseSchema(name="people_names", description="List of people mentioned."),
    ResponseSchema(name="places_names", description="List of places mentioned."),
    ResponseSchema(name="other_companies_referred", description="Other companies mentioned."),
    ResponseSchema(name="related_industries", description="Industries or sectors referenced."),
    ResponseSchema(name="market_implications", description="Implications for stock market or economy."),
    ResponseSchema(name="confidence_score", description="Score between 0 and 1 for model confidence.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt_template = PromptTemplate(
    template="""
You are a financial analyst LLM.
Analyze the following company news and return the result as a structured JSON.

Company: {company_name}
Stock Code: {stock_code}
News Summary: {news}

{format_instructions}
""",
    input_variables=["company_name", "stock_code", "news"],
    partial_variables={"format_instructions": format_instructions}
)

chain = LLMChain(llm=llm, prompt=prompt_template)
#result = chain.run(company_name=company_name, stock_code=stock_code, news=news_summary)


chain = prompt_template | llm
 
response = chain.invoke({"company_name": company_name, "stock_code": stock_code, "news": news_summary}, config={"callbacks": [langfuse_handler]})

final_output = output_parser.parse(response.content)

final_output