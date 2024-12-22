prompt_template = """ 
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
You do this by asking questions about the text below:
Create Only 10 Questions that will prepare the coders or programmers for their exams and coding tests. Make sure only 10 questions are created.

--------------
{text}
--------------


Make sure not to loose any important information from the text.

Questions: 
"""

refine_template = """
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
We have received some practice questions to a certain extent: {existing_answer}
We have the option to refine the existing questions or create new ones.
(only if necessary) with some more context below.Given the new context, refine the original questions in English. Make sure create only 10 Questions not more than that.

--------------
{text}
--------------


If the context is not helpful, please provide the original questions.

Questions:
"""
