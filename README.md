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

<a href="https://drive.google.com/drive/folders/16GvZWfqZREYVqnP9G2zgB55LAOyJUx-d?usp=sharing" target="_blank">Repository</a> for the processed data. <a href="https://drive.google.com/drive/folders/1E5r_OoTsZrQtrZeA4DpglN0ya3roGDt0?usp=sharing" target="_blank">Folder</a> for data information. Number of articles by <a href="/__resources__/hourly_count.pdf" target="_blank">hour</a>, <a href="__resources__/daily_count.pdf" target="_blank">date</a>, and <a href="__resources__/yearly_count.pdf" target="_blank">year</a>. Number of stocks mentioned from *Jan 2015* to *Jul 2019*. 

![alt text](./__resources__/stock_count.jpg?raw=true "Title")

Dictionary of parameters: https://github.com/xiubooth/Textual-Analysis/blob/main/params/params.py

## SSESTM
<a href="https://drive.google.com/drive/folders/1EJzldxDb6OJwT19V1o_R27hMuDxfb3MI?usp=sharing" target="_blank">Repository</a> for trained models, hyper-parameters, and L/S portfolio positions, weights and returns

|                       |  Long-EW  |  Short-EW  |  LS-EW  |  Long-VW  |  Short-VW  |  LS-VW  |  Index  |
|-----------------------|:---------:|:----------:|:-------:|:---------:|:----------:|:-------:|:-------:|
| Average daily return  | 0.264%    | 0.159%     | 0.423%  | 0.309%    | 0.106%     | 0.416%  | -0.001% |
| Sharpe ratio          | 1.72      | 1.26       | 11.01   | 2.12      | 1.06       | 8.68    | 0.07    |
| Turnover              | 66.6%     | 70.5%      | 68.6%   | 65.9%     | 68.7%      | 67.3%   | /       |

![alt text](./__resources__/backtest_ssestm.jpg?raw=true "Title")

[//]: # (#### Sentiment Analysis)

[//]: # (<a href="/__resources__/sentiment.xlsx" target="_blank">Full summary</a> of results for analysis of sentiment charged words.)

[//]: # ()
[//]: # (**Top 10 frequent words:** `涨&#40;0.0619&#41;`, `跌&#40;0.0536&#41;`, `发展&#40;0.0433&#41;`, `胜&#40;0.0241&#41;`, `平安&#40;0.0216&#41;`, `高新&#40;0.0184&#41;`, `建设&#40;0.0177&#41;`, `动力&#40;0.0167&#41;`, `健康&#40;0.0158&#41;`, `创业&#40;0.0135&#41;`)

[//]: # ()
[//]: # (**Bottom 10 sentiment words:** `跌&#40;-0.0536&#41;`, `垃圾&#40;-0.0071&#41;`, `杀&#40;-0.0032&#41;`, `反弹&#40;-0.0027&#41;`, `下跌&#40;-0.0027&#41;`, `差&#40;-0.0017&#41;`, `机会&#40;-0.0016&#41;`, `大跌&#40;-0.0014&#41;`, `问题&#40;-0.0014&#41;`, `受益&#40;-0.0013&#41;`)

[//]: # ()
[//]: # (**Top 10 sentiment words:** `涨&#40;0.0233&#41;`, `发展&#40;0.0109&#41;`, `胜&#40;0.0064&#41;`, `创业&#40;0.0062&#41;`, `建设&#40;0.0060&#41;`, `高新&#40;0.0058&#41;`, `健康&#40;0.0049&#41;`, `幸福&#40;0.0044&#41;`, `创新&#40;0.0036&#41;`, `动力&#40;0.0036&#41;`)

## Doc2Vec
<a href="https://drive.google.com/drive/folders/1E154z82RoUGKTYgvtNx11R7tT6eGfFuq?usp=sharing" target="_blank">Repository</a> for trained models, hyper-parameters, and L/S portfolio positions, weights and returns

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

[//]: # (- <a href="https://drive.google.com/drive/folders/1DwefHc4P4FTRb9HV3UMXN2Jh4XVNyo-Y?usp=sharing" target="_blank">A simple strategy</a> that counts the occurrence of `涨` and `跌` &#40;which have the highest occurrence and the most positive/negative sentiments&#41;)

## TODO
- Look at the performance of the strategies on the CSI 300 constituent stocks.
