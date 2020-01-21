### How to run the server:

#### Download the following font embeddings and put them into `\neural`:
https://drive.google.com/open?id=1J-yurhCBoT4TQhraIL65TkY-fqXZtQMc

### Or compute them yourself by running:
```bash
$ python ./neural/compute_embeddings.py
```

### For an optimized version you could try building the Ball Tree for the font embedding space:
```bash
$ python ./neural/build_ball_tree.py
```

#### Run the server:
```bash
$ pipenv install
$ pipenv run python main.py
```
