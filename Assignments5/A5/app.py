from flask import Flask, request, render_template
import time
import re
from models.Elastic_Pr import IndexerWithPR
from models.Pr import Pr
from models.TfIdf_Ranker import TfIdfRanker

app = Flask(__name__)

# Initialize IndexerWithPR and TfIdfRanker
es_pr_handler = IndexerWithPR()
es_pr_handler.run_indexer()

tfidf_handler = TfIdfRanker()
tfidf_handler.run_indexer()

def emphasize_query(content, query, context_size=50, snippet_limit=3):
    """Highlights query terms in content and returns snippets."""
    pattern = re.compile(rf'\b{re.escape(query)}\b', re.IGNORECASE)
    found_matches = list(pattern.finditer(content))

    if not found_matches:
        return content[:200] + "..."

    highlighted_snippets = []
    for match in found_matches[:snippet_limit]:
        snippet_start = max(0, match.start() - context_size)
        snippet_end = min(len(content), match.end() + context_size)
        snippet_text = content[snippet_start:snippet_end]
        formatted_snippet = pattern.sub(r'<b>\g<0></b>', snippet_text)
        highlighted_snippets.append(formatted_snippet)

    return " ... ".join(highlighted_snippets) + ("..." if len(highlighted_snippets) == snippet_limit else "")

@app.route('/')
def homepage():
    """Renders the homepage."""
    return render_template('index.html')

@app.route('/search_es_pr', methods=['GET'])
def search_es_pr():
    """Handles search queries, fetches and aggregates results from Elasticsearch with PageRank and TF-IDF."""
    start_time = time.time()
    query_text = request.args.get('query', '')
    print(f"Search query: {query_text}")

    # Elasticsearch with PageRank results
    es_response = es_pr_handler.es_client.search(
        index='simple',
        source_excludes=['url_lists'],
        size=100,
        query={
            "script_score": {
                "query": {"match": {"text": query_text}},
                "script": {"source": "_score * doc['pagerank'].value"}
            }
        }
    )

    es_search_results = [
        {
            'title': emphasize_query(hit["_source"]['title'], query_text),
            'url': hit["_source"]['url'],
            'text': emphasize_query(hit["_source"]['text'], query_text),
        }
        for hit in es_response['hits']['hits']
    ]

    # Elasticsearch with TF-IDF results
    tfidf_response = tfidf_handler.es_client.search(
        index='extend',
        source_excludes=['url_lists'],
        size=100,
        query={"match": {"text": query_text}}
    )

    tfidf_search_results = [
        {
            'title': emphasize_query(hit["_source"]['title'], query_text),
            'url': hit["_source"]['url'],
            'text': emphasize_query(hit["_source"]['text'], query_text),
        }
        for hit in tfidf_response['hits']['hits']
    ]

    # Aggregating results from both Elasticsearch queries
    aggregated_results = es_search_results + tfidf_search_results
    print(f"Aggregated results: {aggregated_results}")
    aggregated_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    total_results = len(aggregated_results)
    end_time = time.time()

    return render_template(
        'search_results.html',
        query=query_text,
        pr_results=es_search_results,
        tfidf_results=tfidf_search_results,
        total_hit=total_results,
        elapse=end_time - start_time
    )

if __name__ == '__main__':
    app.run(debug=True)
