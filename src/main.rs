use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder,SentenceEmbeddingsModel};
use tch::Device;
use pgvector::Vector;
use postgres::{Client, NoTls};
use std::iter::zip;

fn process_embeddings(model: &SentenceEmbeddingsModel, client: &mut Client) -> anyhow::Result<usize> {
    let mut sentences: Vec<String> = Vec::new();
    let mut ids: Vec<i64> = Vec::new();

    // fetch 1000 at a time; i tried 10000 and the
    // GPU crapped out
    for row in client.query("SELECT id, content FROM chunks WHERE vectored = false LIMIT 100", &[])? {
        let id: i64 = row.get("id");
        ids.push(id);
        sentences.push(row.get("content"));
    }

    let borrowed_sentences: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    if borrowed_sentences.len() > 0 {
        println!("updating {} sentences", borrowed_sentences.len());        
        let embeddings = model.encode(&borrowed_sentences)?;
        let insertables = zip(embeddings, ids);
        for i in insertables {
            let embedding = Vector::from(i.0);
            let res = client.query("UPDATE chunks SET embedding = $1, vectored = true WHERE id = $2", &[&embedding, &i.1]);
            match res {
                Ok(_) => {},
                Err(e) => println!("error parsing header: {e:?}"),
            }
        }
    }
    return Ok(borrowed_sentences.len())
}

fn main() -> anyhow::Result<()> {
    let mut client = Client::configure()
        .host("localhost")
        .dbname("postgres")
        .connect(NoTls)?;

    client.execute("CREATE EXTENSION IF NOT EXISTS vector", &[])?;

    let model = SentenceEmbeddingsBuilder::local("models/intfloat/e5-large-v2")
        .with_device(Device::Mps)
        .create_model()?;

    let mut count = 0;
    loop {
        let result = process_embeddings(&model, &mut client)?;
        count = count + result;
        println!("processed: {}", count);
        if result == 0 {
            break;
        }
    }
    Ok(())
}
