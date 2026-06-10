import requests
import pandas as pd
import re


class UniverseBuilder:
    def __init__(self):
        self.universe = set()

    def _clean_ticker(self, ticker: str):
        if not isinstance(ticker, str):
            return None

        t = ticker.strip().upper()
        t = t.replace(".", "-")

        # remove footnotes like BRK.B[1]
        t = re.sub(r"\[.*?\]", "", t)

        if not re.match(r"^[A-Z0-9\-]{1,10}$", t):
            return None

        return t

    def _add_tickers(self, tickers):
        for t in tickers:
            clean = self._clean_ticker(t)
            if clean:
                self.universe.add(clean)

    def _fetch_table(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return pd.read_html(r.text)

    def get_sp500(self):
        tables = self._fetch_table(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        return tables[0]["Symbol"].tolist()

    def get_nasdaq100(self):
        tables = self._fetch_table(
            "https://en.wikipedia.org/wiki/Nasdaq-100"
        )

        for t in tables:
            cols = [c.lower() for c in t.columns]

            if "ticker" in cols:
                return t[t.columns[cols.index("ticker")]].tolist()

            if "symbol" in cols:
                return t[t.columns[cols.index("symbol")]].tolist()

        return []

    def build(self):
        print("Fetching S&P 500...")
        self._add_tickers(self.get_sp500())

        print("Fetching Nasdaq-100...")
        self._add_tickers(self.get_nasdaq100())


        
        return sorted(self.universe)

    def to_dataframe(self):
        return pd.DataFrame(sorted(self.universe), columns=["ticker"])

    def save_csv(self, path="universe.csv"):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} tickers to {path}")


if __name__ == "__main__":
    builder = UniverseBuilder()

    universe = builder.build()

    df = builder.to_dataframe()

    print("\nTotal clean tickers:", len(universe))
    print(df.head(20))

    builder.save_csv()

    print("\nDone.")
    print(r.status_code)
    print(r.text[:1000])