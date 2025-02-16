import json
import pandas as pd
import numpy as np
from pathlib import Path


class Pr:

    def __init__(self, alpha):
        # Set the path to the crawled folder and alpha value for PageRank calculation
        self.crawled_folder = Path("resources/crawled")
        self.alpha = alpha

    def url_extractor(self):
        # Initialize empty dictionaries to hold URL mappings and all URLs
        url_maps = {}
        all_urls = set()

        # Use pathlib to iterate over files in the folder and read the JSON content
        for file in self.crawled_folder.glob("*.txt"):
            with open(file, 'r', encoding='utf-8') as f:
                j = json.load(f)
            all_urls.add(j['url'])
            # Add linked URLs from the 'url_lists' field
            for s in j['url_lists']:
                all_urls.add(s)
            # Store the URL mappings
            url_maps[j['url']] = list(set(j['url_lists']))

        all_urls = list(all_urls)
        return url_maps, all_urls

    def pr_calc(self):
        # Extract URL mappings and all URLs
        url_maps, all_urls = self.url_extractor()

        # Create a DataFrame for the URL matrix
        url_matrix = pd.DataFrame(columns=all_urls, index=all_urls)

        # Calculate PageRank for each URL based on its links
        for url in url_maps:
            if url_maps[url]:  # Ensure there are linked URLs
                # Update the URL matrix with the initial rank values
                url_matrix.loc[url] = (1 - self.alpha) * (1 / len(all_urls))
                # Update the matrix for linked URLs based on the alpha factor
                url_matrix.loc[url, url_maps[url]] = url_matrix.loc[url, url_maps[url]] + (
                            self.alpha * (1 / len(url_maps[url])))

        # Set default values for URLs with no links
        url_matrix.loc[url_matrix.isnull().all(axis=1), :] = (1 / len(all_urls))

        # Initial vector for PageRank (uniform distribution)
        x0 = np.matrix([1 / len(all_urls)] * len(all_urls))
        P = np.asmatrix(url_matrix.values)

        # Iterate to calculate the final PageRank until convergence
        prev_Px = x0
        Px = x0 * P
        iterations = 0
        while np.any(abs(np.asarray(prev_Px).flatten() - np.asarray(Px).flatten()) > 1e-8):
            iterations += 1
            prev_Px = Px
            Px = Px * P

        # Print the result of the PageRank calculation after convergence
        print(f'Converged in {iterations} iterations: {np.around(np.asarray(Px).flatten().astype(float), 5)}')

        # Store the PageRank result in a DataFrame
        self.pr_result = pd.DataFrame(Px, columns=url_matrix.index, index=['score']).T
