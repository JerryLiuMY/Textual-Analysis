# Textual-Analysis
<p align="left">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
</p>

## Data Information
|                                                     |  Remaining Sample Size  |  Observations Removed  |
|-----------------------------------------------------|:-----------------------:|:----------------------:|
| Total number of articles                            | 43,132,204              |                        |
| Drop articles out of date range                     | 37,256,210              | 5,875,994              |
| Drop articles with no associated stock              | 33,062,855              | 4,193,355              |
| Drop articles with more than one stock tagged       | 32,143,209              | 919,646                |
| Drop articles without match with the CSMAR database | 22,543,726              | 9,599,483              |

Dictionary of parameters: https://github.com/xiubooth/Textual-Analysis/blob/main/params/params.py

<a href="https://drive.google.com/drive/folders/1ucT05lVyNUZ93czPlQT-B4naFscslXkF?usp=sharing" target="_blank">Repository</a> for data information. Number of articles by <a href="/__resources__/hourly_count.pdf" target="_blank">hour</a>, <a href="__resources__/daily_count.pdf" target="_blank">date</a>, and <a href="__resources__/yearly_count.pdf" target="_blank">year</a>. Number of stocks mentioned from *Jan 2015* to *Jul 2019*. 

![alt text](./__resources__/stock_count.jpg?raw=true "Title")

## SSESTM
<a href="https://drive.google.com/drive/folders/1mxMuPzGZvH_qNtP9J_t92lzjrZX5jdEu?usp=sharing" target="_blank">Repository</a> for trained models, hyper-parameters, and L/S portfolio positions, weights and returns

|                       |  Long-EW  |  Short-EW  |  LS-EW  |  Long-VW  |  Short-VW  |  LS-VW  |  Index  |
|-----------------------|:---------:|:----------:|:-------:|:---------:|:----------:|:-------:|:-------:|
| Average daily return  | 0.264%    | 0.159%     | 0.423%  | 0.309%    | 0.106%     | 0.416%  | -0.001% |
| Sharpe ratio          | 1.72      | 1.26       | 11.01   | 2.12      | 1.06       | 8.68    | 0.07    |
| Turnover              | 66.6%     | 70.5%      | 68.6%   | 65.9%     | 68.7%      | 67.3%   | /       |

![alt text](./__resources__/backtest_ssestm.jpg?raw=true "Title")

#### Sentiment Analysis
<a href="/__resources__/sentiment.xlsx" target="_blank">Full summary</a> of results for analysis of sentiment charged words.

**Top 10 frequent words:** `涨(0.0619)`, `跌(0.0536)`, `发展(0.0433)`, `胜(0.0241)`, `平安(0.0216)`, `高新(0.0184)`, `建设(0.0177)`, `动力(0.0167)`, `健康(0.0158)`, `创业(0.0135)`

**Bottom 10 sentiment words:** `跌(-0.0536)`, `垃圾(-0.0071)`, `杀(-0.0032)`, `反弹(-0.0027)`, `下跌(-0.0027)`, `差(-0.0017)`, `机会(-0.0016)`, `大跌(-0.0014)`, `问题(-0.0014)`, `受益(-0.0013)`

**Top 10 sentiment words:** `涨(0.0233)`, `发展(0.0109)`, `胜(0.0064)`, `创业(0.0062)`, `建设(0.0060)`, `高新(0.0058)`, `健康(0.0049)`, `幸福(0.0044)`, `创新(0.0036)`, `动力(0.0036)`

## Doc2Vec
<a href="https://drive.google.com/drive/folders/1IOhDnHQy8OIk4FSGJsn3ks5Fe4ktauzj?usp=sharing" target="_blank">Repository</a> for trained models, hyper-parameters, and L/S portfolio positions, weights and returns

|                       |  Long-EW  |  Short-EW  |  LS-EW  |  Long-VW  |  Short-VW  |  LS-VW  |  Index  |
|-----------------------|:---------:|:----------:|:-------:|:---------:|:----------:|:-------:|:-------:|
| Average daily return  | 0.059%    | 0.234%     | 0.293%  | -0.044%   | 0.268%     | 0.224%  | -0.001% |
| Sharpe ratio          | 0.63      | 1.52       | 6.92    | -0.60     | 1.80       | 3.09    | 0.07    |
| Turnover              | 86.4%     | 91.5%      | 89.0%   | 90.4%     | 92.4%      | 91.4%   | /       |

![alt text](./__resources__/backtest_doc2vec.jpg?raw=true "Title")


## BERT
|                       |  Long-EW  |  Short-EW  |  LS-EW  |  Long-VW  |  Short-VW  |  LS-VW  |  Index  |
|-----------------------|:---------:|:----------:|:-------:|:---------:|:----------:|:-------:|:-------:|
| Average daily return  | 0.005%    | 0.307%     | 0.312%  | -0.047%   | 0.231%     | 0.183%  | -0.001% |
| Sharpe ratio          | 0.17      | 1.77       | 6.81    | -0.67     | 1.65       | 2.70    | 0.07    |
| Turnover              | 88.3%     | 91.5%      | 89.9%   | 91.3%     | 92.9%      | 92.1%   | /       |

![alt text](./__resources__/backtest_bert.jpg?raw=true "Title")

## Caveats
- The articles whose `returns` or `three-day returns` cannot be obtained are removed from the data during pre-processing (many of which corresponds to *delisted* stocks). This corresponds to purposefully avoiding future **delisting** -- the information that we could not know in advance.
- It is not feasible to directly short stocks in the Chinese market. We can short on ETFs instead.
- A simple strategy that counts the occurrence of `涨` and `跌` (which have the highest occurrence and the most positive/negative sentiments)

## TODO
- Look at the performance of the strategies on the CSI 300 constituent stocks.
