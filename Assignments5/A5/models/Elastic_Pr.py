import json
import pickle
from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from models.Pr import Pr


class IndexerWithPR:
    def __init__(self):
        self.crawled_folder = Path('resources/crawled')
        self.indexed_data_path = Path("indexed_data.pkl")

        # Load URL mapping
        with open(self.crawled_folder / 'url_list.pickle', 'rb') as f:
            self.file_mapper = pickle.load(f)

        # Connect to Elasticsearch
        self.es_client = Elasticsearch(
            "https://localhost:9200",
            basic_auth=("elastic", "8IYVqmEXvxJuPAmZs=46"),
            ca_certs="~/http_ca.crt"
        )

        # Load PageRank instance
        with open('pickled/pr_instance.pkl', 'rb') as f:
            self.pr = pickle.load(f)

    def run_indexer(self):
        """Indexes data to Elasticsearch, using a cache to avoid redundant processing."""
        # Create Elasticsearch index (delete and recreate if needed)
        self.create_elasticsearch_index()

        # Load indexed data from pickle if available, otherwise process files
        indexed_data = self.load_or_process_indexed_data()

        # Prepare documents for bulk indexing
        actions = self.prepare_bulk_actions(indexed_data)

        # Use Elasticsearch's bulk helper function to index all documents at once
        self.bulk_index_documents(actions)

    def create_elasticsearch_index(self):
        """Deletes and creates the Elasticsearch index."""
        self.es_client.options(ignore_status=[400, 404]).indices.delete(index='simple')
        self.es_client.options(ignore_status=[400]).indices.create(index='simple')
        print("Elasticsearch index created")

    def load_or_process_indexed_data(self):
        """Loads indexed data from pickle file or processes new files."""
        if self.indexed_data_path.exists():
            print("Loading indexed data from pickle file...")
            with open(self.indexed_data_path, "rb") as f:
                indexed_data = pickle.load(f)
        else:
            print("Processing files and creating indexed data...")
            indexed_data = self.process_crawled_files()
            # Save processed data to pickle file for future use
            with open(self.indexed_data_path, "wb") as f:
                pickle.dump(indexed_data, f)
        return indexed_data

    def process_crawled_files(self):
        """Processes crawled files and computes PageRank scores."""
        indexed_data = []
        for file in self.crawled_folder.glob("*.txt"):
            with open(file, 'r', encoding='utf-8') as f:
                j = json.load(f)
            j['id'] = j['url']
            j['pagerank'] = self.pr.pr_result.loc[j['id']].score
            indexed_data.append(j)
        return indexed_data

    def prepare_bulk_actions(self, indexed_data):
        """Prepares actions for bulk indexing."""
        actions = [
            {
                "_op_type": "index",
                "_index": "simple",
                "_id": doc["id"],
                "_source": doc
            }
            for doc in indexed_data
        ]
        return actions

    def bulk_index_documents(self, actions):
        """Performs bulk indexing using Elasticsearch's bulk helper function."""
        print("Sending bulk index to Elasticsearch...")
        success, failed = bulk(self.es_client, actions)
        print(f"Bulk indexing complete. {success} documents indexed, {failed} failed.")