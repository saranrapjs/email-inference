package main

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"fmt"
	netmail "net/mail"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"unicode/utf8"

	"github.com/k3a/parsemail"
	_ "github.com/lib/pq"

	// "github.com/pkoukk/tiktoken-go"
	// "github.com/nlpodyssey/cybertron/pkg/tasks"
	// "github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/saranrapjs/maildotapp"
	// "github.com/pgvector/pgvector-go"
	// "github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
)

func loadTokenizer() (func(text string) []string, error) {
	vocab, err := vocabulary.NewFromFile(filepath.Join("models/intfloat/e5-large-v2/vocab.txt"))
	if err != nil {
		return nil, fmt.Errorf("failed to load vocabulary for question-answering: %w", err)
	}
	t := wordpiecetokenizer.New(vocab)
	return func(text string) []string {
		cls := wordpiecetokenizer.DefaultClassToken
		sep := wordpiecetokenizer.DefaultSequenceSeparator
		// TODO: this is hardcoded but should be
		// derived from the model's tokenizer config
		text = strings.ToLower(text)
		return append([]string{cls}, append(tokenizers.GetStrings(t.Tokenize(text)), sep)...)
	}, nil
}

var tokenize = func(text string) []string {
	return nil
}

func main() {
	f, err := os.Create("./pprof")
	if err != nil {
		panic(err)
	}
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()

	ctx := context.Background()

	tokenize, err = loadTokenizer()
	if err != nil {
		panic(err)
	}
	db, err := sql.Open("postgres", "postgres://127.0.0.1:5432/postgres?sslmode=disable")
	if err := resetDB(db); err != nil {
		panic(err)
	}
	mboxes, err := maildotapp.NewMailboxes()
	if err != nil {
		panic(err)
	}
	defer mboxes.Close()

	fetchMailbox := func(mbox maildotapp.Mailbox) {
		query := mboxes.Query(maildotapp.MailboxQuery{
			Mailbox:      mbox,
			BatchResults: 1000,
		})
		var batchCount int
		var count int
		for {
			messages, err := query()
			if len(messages) == 0 {
				break
			}
			if err != nil {
				panic(err)
			}
			for _, msg := range messages {
				r, err := msg.Open()
				// rarely, a file pointer is missing
				// from disk...
				if err != nil {
					continue
				}
				e, err := parsemail.Parse(r)
				// we ignore parsing errors, counting on them to
				// be few...
				if err != nil && len(e.TextBody) == 0 {
					continue
				}
				if err := insertChunks(db, ctx, e); err != nil {
					panic(err)
				}
				count++
			}
			batchCount += len(messages)
			fmt.Printf("chunked %v out of %v emails\n", count, batchCount)
			if batchCount >= 10000 {
				break
			}
		}
	}

	inbox, err := mboxes.Mailbox("jeff@bigboy.us", maildotapp.Inbox)
	if err != nil {
		panic(err)
	}
	fetchMailbox(inbox)
	sent, err := mboxes.Mailbox("jeff@bigboy.us", "Sent Items")
	if err != nil {
		panic(err)
	}
	fetchMailbox(sent)
}

func fromToString(list []*netmail.Address) string {
	var out string
	for i, m := range list {
		if i != 0 {
			out += ", "
		}
		out += m.Name + " (" + m.Address + ")"
	}
	return out
}

const (
	nullRune  = '\x00'
	quoteRune = '>'
)

func sanitizeUTF8(s string) string {
	valid := make([]rune, 0, len(s))
	for i, r := range s {
		if len(s) > i+1 && rune(s[i+1]) == quoteRune {
			continue
		}
		if utf8.ValidRune(r) && r != nullRune {
			valid = append(valid, r)
		}
	}
	return string(valid)
}

func insertChunks(db *sql.DB, ctx context.Context, email parsemail.Email) error {
	insertStmt, err := db.Prepare("INSERT INTO chunks (emailSubject, emailDate, emailFrom, emailTo, nchunk, content) VALUES ($1, $2, $3, $4, $5, $6)")
	if err != nil {
		return err
	}
	body := sanitizeUTF8(email.TextBody)
	chunks, err := breakToChunks(body)
	if err != nil {
		return err
	}
	for i, chunk := range chunks {
		if _, err := insertStmt.Exec(sanitizeUTF8(email.Subject), email.Date.String(), sanitizeUTF8(fromToString(email.From)), sanitizeUTF8(fromToString(email.To)), i, "passage: "+chunk); err != nil {
			fmt.Fprintf(os.Stderr, "error on insert %v\n", chunk)
			return err
		}
	}
	return nil
}

func resetDB(db *sql.DB) error {
	if _, err := db.Exec("CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
		return err
	}
	if _, err := db.Exec("DROP TABLE IF EXISTS chunks"); err != nil {
		return err
	}
	// TODO: derive vector length from model config
	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS chunks (
	id bigserial PRIMARY KEY,
	content text,
	vectored boolean DEFAULT false,
	embedding vector(1024),
	emailDate TEXT,
	emailFrom TEXT,
	emailTo TEXT,
	emailSubject TEXT,
	nchunk INTEGER
);`); err != nil {
		return err
	}
	return nil
}

// breakToChunks breaks "in" into chunks of
// approximately chunkSize tokens each, returning the chunks.
func breakToChunks(in string) ([]string, error) {
	chunks := []string{""}
	scanner := bufio.NewScanner(strings.NewReader(in))
	scanner.Split(splitByParagraph)

	for scanner.Scan() {
		chunks[len(chunks)-1] = chunks[len(chunks)-1] + scanner.Text() + "\n"
		toks := tokenize(chunks[len(chunks)-1])
		if len(toks) > 300 {
			chunks = append(chunks, "")
		}
	}

	// If we added a new empty chunk but there weren't any paragraphs to add to
	// it, make sure to remove it.
	if len(chunks[len(chunks)-1]) == 0 {
		chunks = chunks[:len(chunks)-1]
	}

	return chunks, nil
}

// splitByParagraph is a custom split function for bufio.Scanner to split by
// paragraphs (text pieces separated by two newlines).
func splitByParagraph(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if i := bytes.Index(data, []byte("\n")); i >= 0 {
		return i + 1, bytes.TrimSpace(data[:i]), nil
	}
	if atEOF && len(data) != 0 {
		return len(data), bytes.TrimSpace(data), nil
	}
	return 0, nil, nil
}
