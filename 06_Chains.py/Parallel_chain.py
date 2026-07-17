from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template='Generate short and simple notes from following text \n {text}',
    input_variables=['text']
)

template2 =  PromptTemplate(
    template='Generate short five short questions about following text \n {text}',
    input_variables=['text']
)

template3 = PromptTemplate(
    template='merge the provided noted and quiz into single documents \n  notes -> {notes} and {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel(
    {'notes':template1 | model | parser,
     'quiz' : template2 | model | parser }
)

merge_chain = template3 | model | parser 

chain = parallel_chain | merge_chain

text = """The Hindus believed in a cyclic universe related to three other beliefs: (i), time is endless and space has infinite extension; (ii), earth is not the center of the universe; and (iii), laws govern all development, including the creation and destruction of the universe. The Indians believed that there were three types of space, physiological, physical, and infinite space. The infinite space consists of undivided consciousness and everything that is inside and outside. However, finite division of space is where time begins, and the division of time is where all beings were first created. It was believed that there are connections between the physical and the psychological worlds, and an equivalence existed between the outer cosmos and the inner cosmos of the individual. This is expressed in the famous sentence – yat pinḍe tad brahmṇḍe, “as in the body so in the universe”.

The ancient Indians mapped out the outer world or the universe at an altar where Yajurveda listed multiples of ten that reached ten million. The numbers used to count to ten million was used as a reference to show the relation of the planets in the universe to Earth, it was not a relevant scale to the entire universe, therefore backing that they believed the universe to be infinite and endless. The Indians calculated the speed of light to be four thousand four hundred and four (4,404) yojanas per nimesa, or about one hundred eighty six thousand (186,000) miles per second. Ancient Indian beliefs also included the belief that the Earth was created after certain stars, these stars include the Sun, Gemini, Aja, and Kurma. Evidence from the Etymological considerations prove this belief and also points towards the discovery of the twin asses, which in western astrology can be found next to the Cancer constellation as Asellus, Borealis, and Asellus Australis.

The Indian cyclic model assumes the existence of countless island universes, which go through their own periods of development and destruction. The conception of cyclicity is taken to be recursive. For an early exposition of these astronomical and cosmological ideas, one may read al-Bīrūnī's classic history of Indian science, composed in 1030 AD, and for an even earlier, popular, view of Indian ideas, one may consult the Vedantic text called the Yoga Vāsiṣṭha (YV), which at 32,000 shlokas is one of the longest books in world literature.[17]
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()