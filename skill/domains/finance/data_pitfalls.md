# Domain: Finance — Data Pitfalls (Stage 1 + Stage 4 Enrichment)

---

## Critical data-level reproducibility traps in Finance papers

**Survivorship bias** — most financial datasets exclude companies that went bankrupt,
were delisted, or were acquired. A strategy evaluated on a survivorship-biased universe
will appear more profitable than it is. Flag if the paper does not explicitly use a
point-in-time constituent list. Add to `data/README_data.md` with instructions to obtain
delisting returns.

**Point-in-time fundamental data** — Compustat and similar databases report financial
statement data with a knowledge date. Using the filing date rather than the knowledge date
(when the data was publicly available) introduces look-ahead bias. Always flag if the
paper does not specify point-in-time data usage.

**Corporate actions** — stock splits, dividends, mergers, and spin-offs require careful
price adjustment. Unadjusted prices produce incorrect returns. Flag the adjustment
methodology if not stated explicitly.

**Data vintage** — macroeconomic data (GDP, CPI, unemployment) is revised after initial
release. Using revised data to make decisions that would have been made on the original
release is look-ahead bias. For macro strategies, always note whether the paper uses
real-time (vintage) data or final revised data.

**Bid-ask spread** — using mid-prices for all trades ignores the half-spread cost of
market orders. High-frequency and market-making papers are particularly sensitive to this.

**Short-selling constraints** — short positions incur borrowing costs not captured in
return series. If the paper describes a long-short strategy, note whether borrow costs
are modelled and add as an implementation assumption if not.

**Calendar effects and holiday handling** — different markets have different trading
calendars. Using the wrong calendar produces misaligned time series when combining
data across markets. Always specify the trading calendar used.
