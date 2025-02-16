import json
import pickle
from pathlib import Path
from elasticsearch import Elasticsearch, helpers


class TfIdfRanker:
    def __init__(self):
        self.crawled_folder = Path('resources/crawled')
        self.pickle_file = 'indexed_documents.pickle'
        self.es_client = Elasticsearch("https://localhost:9200",
                                       basic_auth=("elastic", "8IYVqmEXvxJuPAmZs=46"),
                                       ca_certs="~/http_ca.crt")
        self.load_dependencies()

    def load_dependencies(self):
        # Load file_mapper for file paths
        with open(self.crawled_folder / 'url_list.pickle', 'rb') as f:
            self.file_mapper = pickle.load(f)

        # Load PageRank instance
        with open('pickled/pr_instance.pkl', 'rb') as f:
            self.pr = pickle.load(f)

    def run_indexer(self):
        indexed_documents = self.load_or_create_index()

        if not indexed_documents:
            indexed_documents = self.index_documents()
            self.save_indexed_documents(indexed_documents)

    def load_or_create_index(self):
        if Path(self.pickle_file).exists():
            return self.load_indexed_documents()
        else:
            self.create_elasticsearch_index()
            return []

    def load_indexed_documents(self):
        with open(self.pickle_file, 'rb') as f:
            indexed_documents = pickle.load(f)
        print(f"Loaded indexed documents from {self.pickle_file}")
        return indexed_documents

    def create_elasticsearch_index(self):
        self.es_client.options(ignore_status=[400, 404]).indices.delete(index='extend')
        self.es_client.options(ignore_status=[400]).indices.create(index='extend')
        print("Elasticsearch index created")

    def index_documents(self):
        indexed_documents = []
        bulk_actions = []

        for file in self.crawled_folder.glob('*.txt'):
            document = self.prepare_document(file)
            indexed_documents.append(document)
            bulk_actions.append(self.create_bulk_action(document))

        if bulk_actions:
            self.perform_bulk_indexing(bulk_actions)

        return indexed_documents

    def prepare_document(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

        doc_data['id'] = doc_data['url']
        doc_data['pagerank'] = self.get_pagerank(doc_data['id'])
        doc_data['tfidf_score'] = self.get_tfidf_score(doc_data['url'])
        doc_data['final_score'] = doc_data['pagerank'] + doc_data['tfidf_score']

        print(f"Prepared {doc_data['url']} with PageRank: {doc_data['pagerank']}, "
              f"TF-IDF: {doc_data['tfidf_score']}, Final Score: {doc_data['final_score']}")

        return doc_data

    def get_pagerank(self, doc_id):
        return self.pr.pr_result.loc[doc_id].score

    def get_tfidf_score(self, url):
        search_result = self.es_client.search(index="extend", body={
            "query": {"match": {"url": url}},
            "explain": True
        })
        if search_result['hits']['hits']:
            return search_result['hits']['hits'][0]['_explanation']['value']
        return 1

    def create_bulk_action(self, document):
        return {
            "_op_type": "index",
            "_index": "extend",
            "_id": document['id'],
            "_source": document
        }

    def perform_bulk_indexing(self, bulk_actions):
        helpers.bulk(self.es_client, bulk_actions)
        print("Bulk indexing completed")

    def save_indexed_documents(self, indexed_documents):
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(indexed_documents, f)
        print(f"Pickled indexed documents to {self.pickle_file}")