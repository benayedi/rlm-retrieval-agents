# Prompt files of the document REPL agent's OBLIQ-Bench runs

Byte-exact copies (SHA-256 in `MANIFEST.json`). Each one is a JSON object of prompt blocks that `src/prompts.py`
applies at import when `GEPA_PROMPTS` names the file; unset, the prompts are unchanged. The text is written by us
(task instructions, output contract); it contains no OBLIQ-Bench query or document text.

| File | Blocks | Runs | Record `prompt_id` | Original location |
|---|---|---|---|---|
| `prompts_math.json` | QUERY_TEMPLATE, LEAN_CLAUSE | MAT-11 | f92043bfd120 | `data/obliq_fd_gemini/` of the original repository |
| `prompts_twitter.json` | QUERY_TEMPLATE, LEAN_CLAUSE | TWI-11 | 299e86ce3e1e | same |
| `prompts_congress.json` | QUERY_TEMPLATE, LEAN_CLAUSE | CON-11 | 891e1319f80c | same |
| `prompts_obliq.json` | QUERY_TEMPLATE | TWQ-2, SP-7 (Qwen3.6, OBLIQ-Twitter) | 00a9b02c72d6 | `data/obliq_twitter/` of the document REPL agent working directory (`FULLDOC_ROOT`) |

**How the file of each run was established.** Every record stores `config.prompt_id`, which `src/prompts.py`
`prologue_id()` computes from the prompt blocks in force. With `GEPA_PROMPTS` set to each file, `prologue_id()` gives
exactly the `prompt_id` in the table, the one in that run's records. The working directory also holds
`prompts_obliq.json.bak`; it gives `669457b02612`, which no TWQ-2 / SP-7 record has, so it did not run.

**Check for benchmark text (2026-11-11).** Every 8-word window of every block (lower-cased, alphanumeric tokens) was
compared with all OBLIQ-Bench queries (Math 151, Twitter 281, Congress 254), all three corpora (3,508 / 72,122 /
213,650 documents) and `analogues/math/final_dataset.json`: no window is shared.
