[![Discord Chat](https://img.shields.io/discord/334891772696330241.svg)](https://discord.gg/ndFR4RF)
[![License](https://img.shields.io/github/license/nikmanG/senti-bot.svg)](LICENSE)
# Senti-Bot
A sentimental analysis bot for Discord.

## Setup
### Docker
TODO: not yet available

### Normal Script
#### Prerequisites:
- Python 3.4.2+
- PostgreSQL
- pip3 (or have pip reference Python 3 installation)

#### Installation
Note: this should be pretty generic among all Linux, MacOS, and Windows systems.

- Move into the directory with `main.py`
- Run `pip3 install -r requirements.txt` (or `pip install -r requirements.txt` depending on Python versions)
- Create a `.env` file in the current directory. It should contain the following variables:
```
TOKEN=<Discord Developer Token>
```

- Run `python3 main.py` (or again depending on install `python main.py` could suffice).<br>
If successful, this should be present on the terminal:
```
Logged in as
Senti-Bot
00000000000000 # This will vary as it is the Bot ID
------
``` 

## Usage
TODO: not yet available

## Credits
Linear SVC, Logisitc Regression, and Naive Bayes taken from [Susan Li](https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Multi%20label%20text%20classification.ipynb).
