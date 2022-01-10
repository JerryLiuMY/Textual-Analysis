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

Dictionaries of parameters: https://github.com/xiubooth/Textual-Analysis/blob/main/params/params.py

## SSESTM
|                       |  Long-EW  |  Short-EW  |  LS-EW  |  Long-VW  |  Short-VW  |  LS-VW  |  Index  |
|-----------------------|:---------:|:----------:|:-------:|:---------:|:----------:|:-------:|:-------:|
| Average daily return  | 0.264%    | 0.159%     | 0.877%  | 0.309%    | 0.106%     | 0.757%  | -0.001% |
| Sharpe ratio          | 1.72      | 1.26       | 11.01   | 2.12      | 1.06       | 8.68    | 0.07    |
| Turnover              | 66.6%     | 70.5%      | 68.6%   | 65.9%     | 68.7%      | 67.3%   | /       |

![alt text](./__resources__/backtest_ssestm.jpg?raw=true "Title")


## Doc2Vec
|                       |  Long-EW  |  Short-EW  |  LS-EW  |  Long-VW  |  Short-VW  |  LS-VW  |  Index  |
|-----------------------|:---------:|:----------:|:-------:|:---------:|:----------:|:-------:|:-------:|
| Average daily return  | 0.059%    | 0.234%     | 0.492%  | -0.044%   | 0.268%     | 0.170%  | -0.001% |
| Sharpe ratio          | 0.63      | 1.52       | 6.92    | -0.60     | 1.80       | 3.09    | 0.07    |
| Turnover              | 86.4%     | 91.5%      | 89.0%   | 90.4%     | 92.4%      | 91.4%   | /       |

![alt text](./__resources__/backtest_doc2vec.jpg?raw=true "Title")

## Caveats
- The articles whose `returns` or `three-day returns` cannot be obtained are removed from the data during pre-processing (many of which corresponds to *delisted* stocks). This corresponds to purposefully avoiding future **delisting** -- the information that we could not know in advance.
- It is not feasible to direct short stocks on the Chinese market. We can short on ETFs instead. 
- Look at performance of the strategies on the CSI 300 constituent stocks.
- A strategy that simply counts the occurrence of `涨/跌` (which have the highest occurrence and the most positive/negative sentiments) could be used as a benchmark for the remaining models.