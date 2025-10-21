# Clara

## Overview

**Clara** is my OpenAI-enabled assistant who can effortlessly answer questions based on the piles of digital notes
that exist in my Obsidian vault. Clara is implemented as a command-line and REPL application that allows you to index, 
manage, and query your local Obsidian Markdown vault using OpenAI’s Vector Store capabilities.

Once your notes are indexed, you can ask natural language questions against them. The app uses OpenAI's `file_search` 
feature to find and reference relevant notes, returning both answers and clickable links back to the original Obsidian 
files.

### Key Features

* 🔍 **Automatic Indexing**: Scans your Obsidian vault for Markdown (`.md`) files and syncs changes with an OpenAI Vector Store.
* 🧠 **Semantic Search**: Query your notes in natural language using OpenAI’s vector search.
* 🔗 **References with Links**: Results include clickable `obsidian://open?path=...` links to open matching notes directly in Obsidian.
* 🧭 **REPL Interface**: Interactive prompt to reindex, query, and explore your notes.
* ⚙️ **CLI Options**: Easily configure your vault path and Vector Store target.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/obsidian-openai-querytool.git
cd obsidian-openai-querytool
```

### 2. Install dependencies using Poetry

```bash
poetry install
```

### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-your-api-key"
```

---

## Usage

### Command Line Indexing

To index your Obsidian vault into a Vector Store (creating one if necessary):

```bash
poetry run python main.py index \
  --vault-root /path/to/your/vault \
  --vector-store-name obsidian-vault
```

If you already have a Vector Store ID:

```bash
poetry run python main.py index \
  --vault-root /path/to/your/vault \
  --vector-store-id vs_abc123
```

---

### REPL Mode

Start the interactive REPL to manage indexing and query your vault:

```bash
poetry run python main.py repl \
  --vault-root /path/to/your/vault \
  --vector-store-id vs_abc123
```

#### REPL Commands

| Command          | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| `index`          | Scans your vault for new or updated files and uploads them. |
| `ask <question>` | Queries your indexed notes using natural language.          |
| `help`           | Displays available commands.                                |
| `exit` / `quit`  | Exits the REPL.                                             |

Example session:

```
clara> index
Uploaded 3 files.

clara> ask What were my key takeaways from the design meeting?
Answer:
- You planned to switch the UX layout to a two-column grid.

References:
- [link to note](obsidian://open?path=/Users/alex/vault/MeetingNotes/design-meeting.md)
```

---

## Configuration Options

| Option                | Description                                                         |
| --------------------- | ------------------------------------------------------------------- |
| `--vault-root`        | Path to your local Obsidian vault (required).                       |
| `--vector-store-id`   | Existing Vector Store ID (optional).                                |
| `--vector-store-name` | Name for a new or existing Vector Store (optional).                 |
| `--link-scheme`       | Choose between `obsidian` (default) or `file` links for references. |

Example:

```bash
poetry run python main.py repl \
  --vault-root ~/Documents/ObsidianVault \
  --vector-store-name MyVault \
  --link-scheme file
```

---

## How It Works

1. **Indexing**: The tool reads all `.md` files in your vault, computes SHA256 hashes, modification times, and uploads only changed or new files.
2. **Vector Storage**: Files are stored in an OpenAI Vector Store with metadata (path, mtime, hash).
3. **Querying**: The REPL uses OpenAI’s `file_search` tool to match semantically similar content.
4. **References**: Each query response includes clickable URIs linking to your original notes.

---

## Troubleshooting

* Ensure `OPENAI_API_KEY` is set in your environment.
* Make sure your vault path is correct and accessible.
* If you get an error creating a Vector Store, verify your OpenAI API plan supports them.

---

## License

MIT License © 2025 Jason Blome
