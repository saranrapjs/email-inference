// Some of this code was derived from code originally written
// by Eli Bendersky here: https://github.com/eliben/code-for-blog/blob/master/2023/go-rag-openai/cmd/rag/rag.go
package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	_ "github.com/lib/pq"
	"github.com/pgvector/pgvector-go"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
)

func main() {
	theQuestion := flag.String("question", "can you tell me anything about the New York Times Tech Guild?", "the question")
	flag.Parse()

	answerQuestion(*theQuestion)
}

const tpl = `{
	    "stream": false,
	    "n_predict": 400,
	    "temperature": 0.0,
	    "stop":
	    [
	        "</s>",
	        "Llama:",
	        "User:"
	    ],
	    "repeat_last_n": 256,
	    "repeat_penalty": 1.18,
	    "top_k": 40,
	    "top_p": 0.5,
	    "tfs_z": 1,
	    "typical_p": 1,
	    "presence_penalty": 0,
	    "frequency_penalty": 0,
	    "mirostat": 0,
	    "mirostat_tau": 5,
	    "mirostat_eta": 0.1,
	    "grammar": "",
	    "n_probs": 0,
	    "image_data":
	    [],
	    "cache_prompt": true,
	    "slot_id": 0,
	    "prompt": %s
	}`

func getChatCompletion(prompt string) ([]byte, error) {
	safePrompt, _ := json.Marshal(prompt)
	asJson := fmt.Sprintf(tpl, safePrompt)
	req, err := http.NewRequest("POST", "http://localhost:8080/completion", strings.NewReader(asJson))
	if err != nil {
		return []byte{}, err
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return []byte{}, err
	}
	resBytes, err := io.ReadAll(res.Body)
	if err != nil {
		return []byte{}, err
	}
	return resBytes, nil
}

// answerQuestion is a scripted interaction with the OpenAI API using RAG.
// It takes a question (constant theQuestion), finds the most relevant
// chunks of information to it and places them in the context for the question
// to get a good answer from the LLM.
func answerQuestion(theQuestion string) {
	ctx := context.Background()
	// textencoding.DefaultModel
	m, err := tasks.Load[textencoding.Interface](&tasks.Config{ModelsDir: "models", ModelName: "intfloat/e5-large-v2"})
	checkErr(err)

	db, err := sql.Open("postgres", "postgres://127.0.0.1:5432/postgres?sslmode=disable")
	checkErr(err)
	defer db.Close()
	_, err = db.Exec("CREATE EXTENSION IF NOT EXISTS vector")
	checkErr(err)
	// SQL query to extract chunks' content along with embeddings.
	stmt, err := db.Prepare(`
		SELECT
			emailSubject,
			emailFrom,
			emailDate,
			content,
			1 - (embedding <=> $1) AS score
		FROM chunks
		ORDER BY
			embedding <=> ($1)
		LIMIT 5
	`)
	checkErr(err)
	defer stmt.Close()
	result, err := m.Encode(ctx, "query: "+theQuestion, int(bert.MeanPooling))
	vector := pgvector.NewVector(result.Vector.Data().F32())

	rows, err := stmt.Query(&vector)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	type scoreRecord struct {
		Subject string
		From    string
		Date    string
		Score   float32
		Content string
	}
	var scores []scoreRecord

	for rows.Next() {
		var (
			subject string
			content string
			from    string
			date    string
			score   float32
		)

		err = rows.Scan(&subject, &from, &date, &content, &score)
		if err != nil {
			log.Fatal(err)
		}

		err = rows.Scan(&subject, &from, &date, &content, &score)
		content = strings.Replace(content, "passage: ", "", -1)
		scores = append(scores, scoreRecord{subject, from, date, score, content})
	}
	if err = rows.Err(); err != nil {
		log.Fatal(err)
	}

	var contextInfo string
	for _, score := range scores {
		fmt.Printf("subject: %s, score:%v\n content: %v\ndate:%v\n", score.Subject, score.Score, score.Content, score.Date)
		contextInfo = contextInfo + fmt.Sprintf("\n\nfrom:%s\ndate:%s\nsubject:%s\nemail:%s", score.From, score.Date, score.Subject, score.Content)
	}
	query := fmt.Sprintf(`This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision. Use the following snippets of emails as context:
%v
User: %v\n\nLlama:`, contextInfo, theQuestion)
	resp, err := getChatCompletion(query)
	checkErr(err)
	type response struct {
		Content string `json:"content"`
	}
	var r response
	err = json.Unmarshal(resp, &r)
	checkErr(err)
	fmt.Println(r.Content)
}

func checkErr(err error) {
	if err != nil {
		panic(err)
	}
}
