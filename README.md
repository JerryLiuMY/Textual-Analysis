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


## SSESTM
|                       |  Long-EW  |  Short-EW  |  LS-EW  |  Long-VW  |  Short-VW  |  LS-VW  |  Index  |
|-----------------------|:---------:|:----------:|:-------:|:---------:|:----------:|:-------:|:-------:|
| Average daily return  | 0.264%    | 0.159%     | 0.877%  | 0.309%    | 0.106%     | 0.757%  | -0.001% |
| Sharpe ratio          | 1.72      | 1.26       | 11.01   | 2.12      | 1.06       | 8.68    | 0.07    |
| Turnover              | 66.6%     | 70.5%      | 68.6%   | 65.9%     | 68.7%      | 67.3%   | /       |

![alt text](./__resources__/backtest.jpg?raw=true "Title")


## Caveats
- The articles whose `returns` or `three-day returns` cannot be constructed are removed from the data during pre-processing (many of which corresponds to *delisted* stocks). This corresponds to purposely avoiding future **delisting** -- the information of which we could not know in advance. 
