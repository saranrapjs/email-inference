# Mail.app + llamafile

This code describes a way to probe emails stored by the MacOS Mail.app locally using a locally running large language model via [llamafile](https://github.com/Mozilla-Ocho/llamafile).

## Prerequisites

This isn't yet working in a way where you can just run a script and it all "works". You'll need at least:

-   postgres and pgvector (via homebrew: `brew install postgresql@14 pgvector`) running at their default port
-   the Go and Rust toolchains
-   an installed copy of the [`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2/tree/main) model downloaded into the `models/` folder
-   a Rust-converted version of that model, using the [`rust-bert` convert_model.py script](https://github.com/guillaume-be/rust-bert#loading-pretrained-and-custom-model-weights)

## Running everything

1. Chunk and store the last 10,000 inbox and sent emails by running — you'll need to modify some of the inbox criteria in `cmd/mail/main.go` as I don't have this parameterized yet:

```shell
go run ./cmd/mail
```

2. Calculate embedding for those ~20,000 emails — this step will be slow, even while it runs on your GPU, because the model's embedding size is 1024. It took me ~2 hours or so to run, but can be paused and restarted as this script will calculate embeddings only for texts that don't yet have them:

```shell
cargo run
```

3. Download and run the `llamafile` server, following [these instructions](https://github.com/Mozilla-Ocho/llamafile#quickstart)

4. Ask a question:

```shell
go run ./cmd/rag --question "your question here"
```

## TODO's

-   Dockerize the postgres bits
-   Script the download/installation/running of llamafile?
-   Parameterize the inbox-specific stuff in `cmd/mail`
