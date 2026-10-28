# The live bots

Each bot is a separate, deterministic program: fixed strategy code plus fixed parameters. No AI model
is consulted while they trade. The same market data and the same settings always produce the same
decisions. All orders and fills are simulated by Atlas's internal paper broker. Nothing is ever sent
to Alpaca or any other broker.

| Name | Code | Parameters | What it does |
|---|---|---|---|
| **Liquidity** | `LiquidityAgent` in `src/atlas/agents/families.py` | `g003-a05` in `live_lineup_v2.json` | Buys only into a very tight spread (≤ 0.35 bps) with at least 1,041 shares offered and bid-side size leaning its way (imbalance ≥ 0.26). Exits when the spread widens 4.7×, when the book flips hard against it, or after about 172 s. Targets about 13.7% of its capital. It may trade rarely: IEX often shows fewer shares than it needs. |
| **Regime** | `RegimeAdaptiveAgent` in `families.py` | `g004-a23` | Uses momentum when short-term volatility is low relative to its longer average (ratio ≤ 0.78), and buys sharp dips (2.7σ over 5 s) when it is high. Small positions (5% of capital), with exits within about 13 s. |
| **MomentumEvo** | `MomentumAgent` in `families.py` | `g004-a12` | Evolved momentum: buys after a +4.35 bps move within 1 s. Exits on any pullback (−0.1 bps) or after about 29 s. Targets about 13.9% of its capital. It is much faster and pickier than the simple Momentum baseline below. |
| **Momentum** | `SimpleMomentumBaseline` in `src/atlas/agents/baselines.py` | fixed defaults | Buys after a +2 bps move over 5 s; exits on −1 bps or after 60 s. |
| **MeanRevert** | `SimpleMeanReversionBaseline` | fixed defaults | Buys after a 2σ drop over 5 s; exits when the move reverts or after 60 s. |
| **Random** | `RandomControl` | fixed | Randomly toggles between flat and a 20% position. This is a control, not a strategy. |
| **TrendFollow** | `TrendFollowAgent` in `src/atlas/agents/slow_families.py` | `g010-a11` | Slow trend follower on 1-minute prices: buys when a ~4-minute moving average is 10.2 bps above a ~13-minute one; exits when the fast average drops 18.5 bps below the slow one, on a 38 bps stop, or after about 65 minutes. Waits 60 minutes between trades. About 19% of its capital. |
| **Pullback** | `PullbackAgent` in `slow_families.py` | `g010-a28` | Buys a 39 bps drop from the 5-minute high and holds for up to about 4.5 hours, aiming for +200 bps, with an 87 bps stop. About 25% of its capital. Trades rarely. |
| **SmartHold** | `SmartHoldAgent` in `slow_families.py` | `g010-a40` | A cautious buy-and-hold: buys from an hour after the open if 1-minute volatility is at most 2 bps; exits when the price is 21 bps below the day's high, volatility rises above 2 bps, on a 19 bps stop, or after about 12 minutes. Waits 3 hours before buying again. 10% of its capital. |
| **Flip** (permanent) | `CoinFlipBaseline` | fixed | Every 5 minutes, flips a fair coin per symbol: hold 1 share or none. |
| **Hold** (permanent) | `BuyAndHoldBaseline` | fixed | Buys an equal-weight basket at the open and holds it. This is the benchmark and is not ranked. |

`live_lineup_v2.json` is the current lineup: ranks 0–2 are the three v1 bots, unchanged, and ranks 3–5 are
the slow bots (the best of each slow family that survived the last walk-forward fold; opening-range and
intraday-momentum bots died out). The walk-forward found **no out-of-sample edge over Hold** for the slow
bots; they are in the lineup to watch. Every slow bot is long-only and flat before the close.

`live_lineup_v1.json` holds the exact frozen parameters (genomes) of the three evolved bots, and
which experiment, generation and config fingerprint they came from. They are three different
strategies: the best liquidity, regime-adaptive and momentum champions of the run. A near-identical
liquidity sibling was deliberately left out. The file's `genome_hash` values match what the paper
reports show for each bot.

These bots were evolved on **synthetic** data, so real-market results show how the system behaves,
not proof of an edge. To change a bot, evolve a new lineup and save it as a new file (for example
`live_lineup_v2.json`). Never edit an existing file, so past results always map to the exact bot
that produced them.
