# Install
``` sh
$ pip install --user pipenv           # add pipenv to $PATH
$ pipenv install                      # install dependencies
```

# Usage
``` sh
$ ./crawl_data.sh                     # download training images
$ pipenv run python src/network.py        # train network
$ pipenv run python src/main.py           # find regions with tools
```
