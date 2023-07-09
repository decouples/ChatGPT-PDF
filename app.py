import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
from sklearn.neighbors import NearestNeighbors


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count
    if end_page is None:
        end_page = total_pages
    else:
        end_page = min(end_page, total_pages)
    text_list = []
    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)
    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('./model')
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'


def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.2,
    )
    comp = completions.choices
    message = comp[0].text
    return message


def generate_answer(file, question, openAI_key):
    load_recommender(file)
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
              "If the search results mention multiple subjects "\
              "with the same name, create separate answers for each. Only include information found in the results and "\
              "don't add any additional information. Make sure the answer is correct and don't output false content. "\
              "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "\
              "search results which has nothing to do with the question. Only answer what is asked. The "\
              "answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "
    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(openAI_key, prompt, "text-davinci-003")
    return answer


# "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "\
recommender = SemanticSearch()


if __name__ == '__main__':
    # import sys
    # file = sys.argv[1]
    # question = sys.argv[2]
    # openAI_key = sys.argv[3]
    # q_answer = generate_answer(file, question, openAI_key)

    question = "Please give the characteristics of the enrolled patients, " \
               "the medication of the experimental group, " \
               "the drug use of the control group, " \
               "the primary endpoint of the clinical trial, " \
               "and the secondary endpoint information of the clinical trial"
    # question = "请给出文章中入组患者特征，实验组用药，对照组用药，临床试验主要终点，临床试验次要终点信息"
    openAI_key = "sk-Qh2WosdfGFGREMzV5asdsafT3BlbkFJN53FuESRG2343RaG9gGLOT"
    pdf_path = r"./path/to/pdf/example.pdf"
    q_answer = generate_answer(pdf_path, question, openAI_key)
