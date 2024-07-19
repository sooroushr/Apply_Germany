

import requests as r
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin
import json
from tavily import TavilyClient
from PyPDF2 import PdfReader
import io
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from llama_index.core.settings import Settings
from langchain_core.output_parsers import JsonOutputParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import PromptTemplate as llama_prompt_template

from llama_index.core import Document as LlamaDocument

from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from langchain_fireworks import ChatFireworks
import os
from llama_index.llms.fireworks import Fireworks


os.environ["FIREWORKS_API_KEY"] = "AAxmKPDxj5HRBGnOVvNTAs0opdjtp7A8RZv1QK6UtrYS6lzR"

fix_url = lambda ref, rel: urljoin(ref, rel) if rel[:4] != "http" else rel
tavily = TavilyClient(api_key="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")
output_parser = JsonOutputParser()
llm_split = ChatFireworks(model="accounts/fireworks/models/mixtral-8x22b-instruct", temperature=0)
# llm = Fireworks(
#     model="accounts/fireworks/models/llama-v3-70b-instruct", api_key="AAxmKPDxj5HRBGnOVvNTAs0opdjtp7A8RZv1QK6UtrYS6lzR"
# )
llm = ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct", temperature=0)


Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")
Settings.llm = llm

class Apply_Extraction:
    def __init__(self):
        pass
        
    def get_tavily_links(self, university, program, keyword):
        query = f"""Masters program {program} {university} {keyword}"""
        print(query)
        response = tavily.search(
            query=query,
            search_depth="advanced",
        )
        return {result['url']: result['title'] for result in response['results']}

    def concurrent_tavily_expanded_search(self, university, program, keywords):
        links = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(self.get_tavily_links, [university]*len(keywords), [program]*len(keywords), keywords):
                for link, title in result.items():
                    if link not in links:
                        links[link] = title
        return links


    def filter_links(self, university, program, links):
        prompt = """
        {links}

        List of some webpages is provided to you in JSON format with link urls as keys and link titles as values.
        University: {university}
        Program: {program}
        Your task is to filter the links that are related to this program and might contain information about one of the following aspects of it:
        Program requirements, Teaching language, Program duration, Application deadline, Language expertise (TOEFL or IELTS grade) or German language level, Grades,
        There might be links to webpages containing aggregated information about one of the mentioned aspects but for all the master programs (For example application deadlines of each program), You must include these links too.

        Provide a JSON output with the same structure as the input and exlcude the non-related links.
        Important rule: Only provide a valid JSON output and nothing else.
        Important rule: The selected links must be from the official university website.
        Important rule: The links must be related to a normal masters program (exluding double degrees or joint programs with other universities or institutions)
        Important rule: You must exlude every link that is not related to any of the aspects.
        """
        return json.loads(llm_split.invoke(prompt.format(links=json.dumps(links), university=university, program=program)).content)



    def is_pdf(self, url, max_retries=3):
        while max_retries > 0:
            try:
                response = r.head(url)
                break
            except:
                max_retries -= 1
                if max_retries == 0:
                    return False
        content_type = response.headers['content-type']
        extension = mimetypes.guess_extension(content_type)
        if not extension:
            return False
        return 'pdf' in extension
    
    
    def get_links(self, link, max_retries=3):
        if self.is_pdf(link):
            return {}
        while max_retries > 0:
            try:
                source = r.get(link).text
                break
            except:
                max_retries -= 1
                if max_retries == 0:
                    return {}
        soup = BeautifulSoup(source)
        links = {}
        for a in filter(lambda a: a.has_attr('href') and a.text.strip(), soup.find_all('a')):
            text = a.text.strip()
            url = fix_url(link, a['href'])
            if url not in links:
                links[url] = text
        return links


    def concurrent_exapnd_links(self, links):
        links = links.copy()

        with ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(self.get_links, links):
                for link, title in result.items():
                    if link not in links:
                        links[link] = title
        return links
    
    def read_link(self, link, max_retries=3):
        print("Reading", link)
        if link[-3:] == 'pdf':
            response = r.get(url=link, timeout=500)
            on_fly_mem_obj = io.BytesIO(response.content)
            pdf_file = PdfReader(on_fly_mem_obj)
            return '\n'.join([page.extract_text() for page in pdf_file.pages])
        # Normal link
        h = html2text.HTML2Text()
        h.ignore_links = True
        while max_retries > 0:
            try:
                source = r.get(link).text
                break
            except:
                max_retries -= 1
                if max_retries == 0:
                    return ""
        text = h.handle(source)
        return text
    
    
    def build_rag(self, links):
        d = 384
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        postproc = MetadataReplacementPostProcessor(
                    target_metadata_key="window"
                )
        # rerank = SentenceTransformerRerank(
        #     top_n = 10,
        #     model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
        # )

        with ThreadPoolExecutor(max_workers=16) as pool:
            documents = [LlamaDocument(text=text) for text in pool.map(self.read_link, links)]

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=4,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

            # Build the index
        nodes = node_parser.get_nodes_from_documents(documents)


        index = VectorStoreIndex(
            nodes,
            storage_context = storage_context,
        )

        query_engine = index.as_query_engine(
        similarity_top_k = 50,
        retriever_mode="all_leaf",
        response_mode='tree_summarize',
        node_postprocessors = [postproc])

        new_summary_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "Find the Exact Answer to the given query. \n"
            "Do not over explain the answer."
            "Do not add here is the answer or here is the link to the answer."
            "Query: {query_str}\n"
            "Answer: "
        )
        new_summary_tmpl = llama_prompt_template(new_summary_tmpl_str)
        query_engine.update_prompts({"response_synthesizer:summary_template": new_summary_tmpl})

        return query_engine
    
    
    def Extract_Aspect(self, university, program):
        initial_links = self.concurrent_tavily_expanded_search(university, program, ["", "application", "admission", "deadline", "requirements"])
        initial_links = self.filter_links(university, program, initial_links)
        expanded_links = self.concurrent_exapnd_links(initial_links)
        expanded_links_filtered = self.filter_links(university, program, expanded_links)
        self.rag = self.build_rag(expanded_links_filtered)
        ask = lambda aspect: self.rag.query("What is the {} of this masters program: {} in {}".format(aspect, program, university)).response
        aspects = ['duration', 'teaching language', 'required german language level', 'required english language level', 'application deadline date', 'minimum required grade']
        
        
        return {aspect: ask(aspect) for aspect in aspects}