from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


def file_processing(file_path):
    loader = PyPDFDirectoryLoader("pdfs")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    print(data)

    question_gen = ""
    for page in data:
        question_gen += page.page_content + " "

    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-4o-mini", chunk_size=10000, chunk_overlap=200
    )

    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=chunk) for chunk in chunk_ques_gen]

    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-4o-mini", chunk_size=1000, chunk_overlap=100
    )

    document_ans_gen = splitter_ans_gen.split_documents(document_ques_gen)
    return document_ques_gen, document_ans_gen


def llm_pipeline(file_path):
    document_ques_gen, document_ans_gen = file_processing(file_path)
    print(document_ques_gen)

    llm_ques_gen_pipeline = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template, input_variables=["text"]
    )
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        template=refine_template, input_variables=["text", "existing_answer"]
    )
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )
    ques = ques_gen_chain.run(document_ans_gen)
    print(ques)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(document_ans_gen, embeddings)
    llm_ans_gen = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    answer_gen_chain = RetrievalQA.from_chain_type(
        llm=llm_ans_gen, chain_type="stuff", retriever=vector_store.as_retriever()
    )
    ques_list = ques.split("\n")
    ques_list = [ques for ques in ques_list if ques]
    return ques_list, answer_gen_chain


def print_answers(ques_list, answer_gen_chain):
    for question in ques_list:
        print("Question: ", question)
        answer = answer_gen_chain.run(question)
        print("Answer: ", answer)
        print("-------------------\n\n")

        # Save answer to file
        with open("answers.txt", "a") as file:
            file.write("Question: " + question + "\n")
            file.write("Answer: " + answer + "\n")
            file.write("\n\n")
