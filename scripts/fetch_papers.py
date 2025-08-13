import feedparser
import os
import requests
import json
from time import sleep
import urllib.parse  # for URL encoding

def fetch_arxiv_papers(query="machine learning", max_results=100, start=0, save_pdf=False):
    base_url = "http://export.arxiv.org/api/query?"
    encoded_query = urllib.parse.quote(query)
    search_query = f"search_query=all:{encoded_query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    url = base_url + search_query
    feed = feedparser.parse(url)

    papers = []
    for entry in feed.entries:
        paper_id = entry.id.split('/abs/')[-1]
        paper = {
            "id": paper_id,
            "title": entry.title.strip().replace('\n', ' '),
            "summary": entry.summary.strip().replace('\n', ' '),
            "authors": [author.name for author in entry.authors],
            "published": entry.published,
            "link": entry.link,
            "pdf_url": next((l.href for l in entry.links if l.type == 'application/pdf'), None),
            "query": query
        }
        papers.append(paper)

        if save_pdf and paper["pdf_url"]:
            pdf_path = f"data/raw_pdfs/{paper_id.replace('/', '_')}.pdf"
            if not os.path.exists(pdf_path):
                print(f"Downloading: {paper['title']}")
                try:
                    pdf = requests.get(paper["pdf_url"], timeout=10)
                    with open(pdf_path, 'wb') as f:
                        f.write(pdf.content)
                except Exception as e:
                    print(f"Failed to download {paper_id}: {e}")
                sleep(1)

    return papers


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw_pdfs", exist_ok=True)

    # AI topics
    queries = [
        "large language models",
        "transformer",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "deep learning"
    ]

    all_papers = {}

    for query in queries:
        for start in range(0, 300, 100):  # paginated: 0, 100, 200
            print(f"\n🔎 Fetching papers for: {query} (start={start})")
            papers = fetch_arxiv_papers(query=query, max_results=100, start=start, save_pdf=True)
            for paper in papers:
                all_papers[paper["id"]] = paper

    # Save metadata
    with open("data/arxiv_papers.json", "w", encoding="utf-8") as f:
        json.dump(list(all_papers.values()), f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(all_papers)} unique papers to data/arxiv_papers.json")