import os
import re
from datetime import datetime
from typing import List, Dict
import time
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from collections import Counter
import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pickle
import joblib

from src.config import OLLAMA_HOST, OLLAMA_MODEL, CONNECTION_STRING, OPT_TRADE_NEWS, COLLECTION_VECTOR, MAX_SIZE_ARTICLE

TRESHOLD = 0.64

class MultilingualNewsRetriever:
    def __init__(self, llm: OllamaLLM):
        self.tfidf_vectorizers = self.load_tfidf_vectorizers()
        # self.model = Ollama(**OPT_TRADE_NEWS)

        self.model = llm
        self.nlp_models = {}
        self.model_lock = threading.Lock()

    def _initialize_vector_store(self):
        self.embeddings = OllamaEmbeddings(**OPT_TRADE_NEWS)
        self.vector_store = PGVector(
            collection_name=COLLECTION_VECTOR,
            connection=CONNECTION_STRING,
            embeddings=self.embeddings,
            use_jsonb=True,
            pre_delete_collection=False,
        )

    def load_tfidf_vectorizers(self):
        vectorizers = {}
        dict_folder = 'learn_dicts'
        for lang in ['en', 'ru', 'fr', 'pt']:
            file_path = os.path.join(dict_folder, f'tfidf_vectorizer_{lang}.joblib')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    # vectorizers[lang] = pickle.load(f)
                    vectorizers[lang] = joblib.load(f)
            else:
                print(f"Warning: TF-IDF vectorizer for {lang} not found.")
        return vectorizers

    def get_nlp_model(self, lang):
        with self.model_lock:
            if lang not in self.nlp_models:
                languages = {
                    'en': 'en_core_web_sm',
                    'ru': 'ru_core_news_sm',
                    'fr': 'fr_core_news_sm',
                    'pt': 'pt_core_news_sm'
                }
                model_name = languages.get(lang)
                if model_name:
                    try:
                        self.nlp_models[lang] = spacy.load(model_name)
                        print(f"Loaded NLP model for {lang}")
                    except OSError:
                        print(f"Model {model_name} not found. Attempting to download...")
                        try:
                            spacy.cli.download(model_name)
                            self.nlp_models[lang] = spacy.load(model_name)
                            print(f"Successfully downloaded and loaded model for {lang}")
                        except Exception as e:
                            print(f"Failed to download model for {lang}: {str(e)}")
                            return None
                else:
                    print(f"Warning: No model name defined for language {lang}")
                    return None
            return self.nlp_models[lang]

    def add_news(self, id: int, source: str, text: str, title: str, created_at: datetime, lang: str) -> None:
        doc = Document(
            page_content=text,
            metadata={
                'id': id,
                "title": title,
                "source": source,
                "created_at": created_at.isoformat(),
                "lang": lang,
            }
        )
        self.vector_store.add_documents([doc])

    def search(self, query_text: str, lang: str, k: int) -> List[Dict]:
        filter_dict = {"lang": {"$eq": lang}}
        docs_and_scores = self.vector_store.similarity_search_with_score(
            query_text,
            k=k,
            filter=filter_dict
        )
        if docs_and_scores:
            return [
            {
                "text": doc.page_content,
                "title": doc.metadata["title"],
                "created_at": datetime.fromisoformat(doc.metadata["created_at"]),
                "lang": doc.metadata["lang"],
                "score": score
            }
            for doc, score in docs_and_scores
        ]
        return None


    def compare_tfidf(self, text1: str, text2: str, lang: str) -> float:
        if lang not in self.tfidf_vectorizers:
            print(f"Warning: TF-IDF vectorizer for {lang} is not available.")
            return 0
        vectorizer = self.tfidf_vectorizers[lang]
        tfidf1 = vectorizer.transform([text1])
        tfidf2 = vectorizer.transform([text2])
        return cosine_similarity(tfidf1, tfidf2)[0][0]

    def compare_entities(self, text1: str, text2: str, lang: str) -> float:
        nlp = self.get_nlp_model(lang)
        if not nlp:
            print(f"Warning: NLP model for {lang} is not available. Skipping entity comparison.")
            return 0
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        entities1 = set([ent.text.lower() for ent in doc1.ents])
        entities2 = set([ent.text.lower() for ent in doc2.ents])
        if not entities1 or not entities2:
            return 0
        return len(entities1.intersection(entities2)) / max(len(entities1), len(entities2))

    def compare_titles(self, title1: str, title2: str, lang: str) -> float:
        nlp = self.get_nlp_model(lang)
        if not nlp:
            print(f"Warning: NLP model for {lang} is not available. Using simple token overlap for title comparison.")
            return self.simple_token_overlap(title1, title2)

        doc1 = nlp(title1)
        doc2 = nlp(title2)

        tokens1 = [token.lemma_.lower() for token in doc1 if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
        tokens2 = [token.lemma_.lower() for token in doc2 if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]

        return self.token_overlap(tokens1, tokens2)

    def simple_token_overlap(self, text1: str, text2: str) -> float:
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        return len(tokens1.intersection(tokens2)) / max(len(tokens1), len(tokens2))

    def token_overlap(self, tokens1: List[str], tokens2: List[str]) -> float:
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)
        intersection = sum((counter1 & counter2).values())
        union = sum((counter1 | counter2).values())
        return intersection / union if union > 0 else 0

    def compare_news(self, new_news: Dict, existing_news: Dict) -> Dict:
        lang = new_news['lang']
        entity_similarity = self.compare_entities(new_news['text'], existing_news['text'], lang)
        title_similarity = self.compare_titles(new_news['title'], existing_news['title'], lang)
        tfidf_similarity = self.compare_tfidf(new_news['text'], existing_news['text'], lang)
        vector_similarity = 1 - existing_news['score'] / 2  # Convert distance to similarity

        total_similarity = (
            0.5 * vector_similarity +
            0.2 * entity_similarity +
            0.2 * title_similarity +
            0.1 * tfidf_similarity
        )

        return {
            "vector_similarity": vector_similarity,
            "entity_similarity": entity_similarity,
            "title_similarity": title_similarity,
            "tfidf_similarity": tfidf_similarity,
            "total_similarity": total_similarity
        }

    def sort_by_total_similarity(self, data_list):
        """
        Сортирует список словарей по полю 'total_similarity' от большего к меньшему.

        Args:
            data_list (list): Список словарей для сортировки.

        Returns:
            list: Отсортированный список словарей.
        """
        return sorted(data_list, key=lambda x: x.get('total_similarity', 0), reverse=True)

    def is_news_similar_old(self, new_news: Dict, threshold: float = TRESHOLD) -> bool:
        start_time = time.time()
        lang = new_news['lang']
        time.sleep(0.1) # Wait for vector store to load
        self._initialize_vector_store()
        similar_news = self.search(new_news['text'], lang, k=200)

        if not similar_news:
            return False, None, None, None, time.time() - start_time
        prompt_template = """
        Вы высококлассный аналитик торговых новостей, в совершенстве владеете русским, английским, португальским и французским языками. Вы способны анализировать, понимать и делать переводы с этих языков на другие языки, умеете критически мыслить и принимать обоснованные решения в строгом соответствии с инструкциями ниже.
        Задача - определить, содержит ли новая новость такое же событие (главную мысль), что и существующая в базе данных.

        Новая новость:
        Заголовок: {new_title}
        Текст: {new_text}

        Существующая новость:
        Заголовок: {existing_title}
        Текст: {existing_text}

        Инструкции для принятия решения:
        - Предварительно проанализируй смысл новой и существующей новости, а не только ключевые слова.
        - Если новая новость содержит такое же событие (главную мысль), что и существующая, то выведи answer = true. Если новая новость содержит отличную от существующей новости событие (главную мысль), то выведи answer = false.
        - КРИТИЧНО: Наличие различий в содержании анализируемых новостей не должны приводить к результату "false".


        Выведи объяснения и комментарии.

        Формат ответа:
        - Ответ должен быть в формате JSON и строго соответствовать структуре ответа: {format_instructions}

        Ответ (строго в формате JSON):
        """

        response_schemas = [
            ResponseSchema(name="answer", description="Ответ: true/false", type="bool"),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["new_title", "new_text", "existing_title", "existing_text"],
            partial_variables={"format_instructions": format_instructions}
        )
        try:
            list_similaritys = []

            for news in similar_news:
                similarities = self.compare_news(new_news, news)
                list_similaritys.append({
                "vector_similarity": similarities['vector_similarity'],
                "entity_similarity": similarities['entity_similarity'],
                "title_similarity": similarities['title_similarity'],
                "tfidf_similarity": similarities['tfidf_similarity'],
                "total_similarity": similarities['total_similarity'],
                    })
            sorted_list = self.sort_by_total_similarity(list_similaritys)
            similarities = sorted_list[0]
            if similarities['total_similarity'] > TRESHOLD:
            # if similarities['vector_similarity'] > 0.90 and similarities['total_similarity'] > 0.90:
                prompt = PROMPT.format(
                    new_title=new_news['title'],
                    new_text=new_news['text'],
                    existing_title=similar_news[0]['title'],
                    existing_text=similar_news[0]['text'],
                )

                response = self.model.invoke(prompt)

                try:
                    parsed_output = parser.parse(response)
                except Exception as e:
                    print('Ошибка при парсинге ответа является ли новость дублем.', str(e))

                if not parsed_output or 'answer' not in parsed_output:
                    return False, None, None, None, time.time() - start_time

                if parsed_output.get('answer', False):
                    return True, similar_news['title'], similarities['vector_similarity'], similarities["total_similarity"], time.time() - start_time
        except Exception as e:
            print('Ошибка при сравнении новостей.', str(e))
        return False, None, None, None, time.time() - start_time

    def is_news_similar(self, new_news: Dict, threshold: float = TRESHOLD) -> bool:
        start_time = time.time()
        lang = new_news['lang']
        self._initialize_vector_store()

        title_news = re.sub(r'^.*?:', '', new_news["title"]).rstrip('.') + ". " + new_news["text"]
        similar_news = self.search(title_news, lang, k=50)

        if not similar_news:
            return False, None, None,  time.time() - start_time

        # prompt_template = """
        # Ты - опытный редактор новостного агентства. Твоя задача - определить, является ли новая новость похожей на уже существующую в базе данных.

        # Новая новость:
        # Заголовок: {new_title}
        # Текст: {new_text}

        # Существующая новость:
        # Заголовок: {existing_title}
        # Текст: {existing_text}

        # Языковые метрики сходства:
        # - Сходство векторов: {vector_similarity}
        # - Сходство сущностей: {entity_similarity}
        # - Сходство заголовков: {title_similarity}
        # - TF-IDF сходство: {tfidf_similarity}

        # Общее взвешенное сходство: {total_similarity}
        # Порог сходства: {threshold}

        # На основе этой информации, считаешь ли ты, что новая новость похожа на существующую? Выведи объяснения и комментарии.

        # Важные указания:
        # 1. Внимательно сравни числовые значения метрик сходства с пороговым значением.
        # 2. Помни, что значения больше порогового считаются превышающими порог.
        # 3. Если значения сходства превышают порог, новости следует считать похожими, даже при наличии небольших различий.
        # 4. Убедись, что твой вывод соответствует числовым данным.
        # 5. Если обнаружишь несоответствие между высокими значениями сходства и выводом о непохожести, перепроверь свою логику.

        # Формат ответа:
        # - Ответ должен быть в формате JSON и строго соответствовать структуре ответа: {format_instructions}

        # Ответ (строго в формате JSON):
        # """
        prompt_template = """
        Вы - опытный редактор новостного агентства со специализацией в выявлении дублирующихся новостей. Ваша задача - определить, сообщают ли две новости об одном и том же событии или факте, даже если они используют разные слова и форматирование.

        Новая новость:
        Заголовок: {new_title}
        Текст: {new_text}

        Существующая новость:
        Заголовок: {existing_title}
        Текст: {existing_text}

        Языковые метрики сходства:
        - Сходство векторов: {vector_similarity} [Показывает семантическое сходство]
        - Сходство сущностей: {entity_similarity} [Показывает совпадение ключевых субъектов/объектов]
        - Сходство заголовков: {title_similarity} [Сходство основной темы]
        - TF-IDF сходство: {tfidf_similarity} [Сходство ключевых слов]

        Общее взвешенное сходство: {total_similarity}
        Порог сходства: {threshold}

        Методология оценки:
        1. Сначала определи ФАКТИЧЕСКОЕ ЯДРО каждой новости:
           - Какое основное событие описывается?
           - Кто основные участники?
           - Где и когда это произошло?
           - Какие ключевые цифры или результаты упоминаются?

        2. Даже если новости используют разные формулировки, словесные обороты, цитаты или дополнительные детали, они считаются ДУБЛЯМИ, если:
           - Описывают то же самое событие/факт
           - Включают тех же главных участников
           - Указывают на то же место и время
           - Передают ту же ключевую информацию

        3. Обратите особое внимание на метрику vector_similarity - это самый надежный показатель семантического сходства.

        4. Числовая оценка:
           - Помните, что значения больше порогового считаются превышающими порог.
           - Если total_similarity > {threshold} ИЛИ vector_similarity > {threshold}, это сильный признак дубликата
           - Если entity_similarity очень высокое (> 0.8), это также указывает на вероятный дубликат даже при умеренных других метриках

        5. ВАЖНО: Новости могут быть дублями даже при наличии дополнительных деталей в одной из них!

        Оцените обе новости и определи, являются ли они ДУБЛЯМИ ПО СМЫСЛУ И ФАКТУ, даже если текстуально они различаются.

        Формат ответа:
        {format_instructions}

        При принятии решения, отдавайте приоритет семантическому сходству (vector_similarity) и сходству сущностей (entity_similarity) над текстуальным сходством.

        Ответ (строго в формате JSON):
        """

        response_schemas = [
            ResponseSchema(name="answer", description="Ответ: true/false", type='bool'),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["new_title", "new_text", "existing_title", "existing_text", "vector_similarity", "entity_similarity", "title_similarity", "tfidf_similarity", "total_similarity", "threshold"],
            partial_variables={"format_instructions": format_instructions}
        )
        try:
            for news in similar_news:
                similarities = self.compare_news(new_news, news)
                if similarities['total_similarity'] < threshold:
                    continue

                prompt = PROMPT.format(
                    new_title=new_news['title'],
                    new_text=new_news['text'],
                    existing_title=news['title'],
                    existing_text=news['text'],
                    **similarities,
                    threshold=threshold
                )

                response = self.model.invoke(prompt)

                try:
                    parsed_output = parser.parse(response)
                except Exception as e:
                    print('Ошибка при парсинге ответа является ли новость дублем.', str(e))

                if not parsed_output or 'answer' not in parsed_output:
                    return False, None, None, None, time.time() - start_time

                if parsed_output["answer"]:
                    return True, news["title"], similarities, time.time() - start_time
        except Exception as e:
            print('Ошибка при сравнении новостей.', str(e))
        return False, None, None, time.time() - start_time