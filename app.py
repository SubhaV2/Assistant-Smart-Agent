import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize LLM and tools
llm = ChatGroq(api_key=st.secrets.get("GROQ_API_KEY"))
search = TavilySearchResults(max_results=2)
parser = StrOutputParser()

# Page Header
st.title("Assistant Smart Agent")
st.markdown("Assistant Agent Powered by Groq.")

# Data Collection/Inputs
with st.form("company_info", clear_on_submit=True):
    product_name = st.text_input("**Product Name** (What product are you selling?):")
    company_url = st.text_input("**Company URL** (The URL of the company you are targeting):")
    product_category = st.text_input("**Product Category** (e.g., 'car/Watch/Cellphone etc'):")
    competitors_url = st.text_area("**Competitors URLs** (e.g., www.xyz.com):")
    value_proposition = st.text_input("**Value Proposition** (Summarize the product’s value):")
    target_customer = st.text_input("**Target Customer** (Name of the target individual):")

    # Submit button
    company_insights = ""
    if st.form_submit_button("Generate Insights"):
        if product_name and company_url:
            with st.spinner("Processing..."):
                # Use search tool to fetch company information
                try:
                    company_information = search.invoke(company_url)
                    st.write(f"Company Data Retrieved: {company_information}")
                except Exception as e:
                    st.error(f"Error fetching company data: {e}")
                    company_information = "No data available."

                # Create dynamic prompt with the updated structure
                prompt = f"""
You are an advanced AI assistant with expertise in business strategy, market analysis, and competitive intelligence.
Using the information provided below, generate a comprehensive analysis focusing on the company’s activities, leadership, competitors, and product strategy.

Details:
- **Product Name**: "{product_name}"
- **Company Website**: "{company_url}"
- **Product Category**: "{product_category}"
- **Competitors URLs**: {competitors_url}
- **Value Proposition**: "{value_proposition}"
- **Target Customer**: "{target_customer}"

Your task is to perform the following analysis:
1. **Company Strategy**:
    - Summarize the company’s activities in the industry relevant to the product being sold.
    - Mention any recent public statements, press releases, or articles where key executives (e.g., Chief Data Officer, Chief Compliance Officer) have discussed relevant topics, such as product strategy, market positioning, or upcoming initiatives.
    - Look for any relevant mentions of technology stack, business focus, or new directions that are being pursued.
    - Optionally, refer to any job postings or skills required in recent job ads, as they may give insights into the company’s strategy.

2. **Competitor Mentions**:
    - Highlight any relevant competitors mentioned in the input, providing context for their relationship with the target company (e.g., direct competitors, substitutes, market position).

3. **Leadership Information**:
    - Identify key leadership figures at the company (e.g., CEO, CTO, or other executives) and their relevance to the company's strategy.
    - If available, include insights from recent press releases, public statements, or articles where they discuss the company's direction, innovations, or challenges.

4. **Product/Strategy Summary**:
    - If the company is publicly traded, summarize insights from their 10-Ks, annual reports, investor presentations, or other public filings that provide information on their strategy, product roadmap, or market positioning.
    - Highlight any strategic initiatives or market trends the company is pursuing as indicated by public documents.

5. **Article Links**:
    - Include links to full articles, press releases, or other sources where key information was gathered from. Make sure to include specific URLs for any relevant external resources.

Ensure the analysis is well-structured, actionable, and formatted clearly for readability.
"""

                # Prompt Template
                prompt_template = ChatPromptTemplate([("system", prompt)])

                # Chain
                chain = prompt_template | llm | parser

                # Generate Insights
                try:
                    company_insights = chain.invoke({
                        "company_information": company_information,
                        "product_name": product_name,
                        "competitors_url": competitors_url,
                        "product_category": product_category,
                        "value_proposition": value_proposition,
                        "target_customer": target_customer
                    })
                except Exception as e:
                    st.error(f"Error generating insights: {e}")
                    company_insights = "Failed to generate insights."

# Display the result
if company_insights:
    st.markdown("### Generated Insights:")
    st.markdown(company_insights)
