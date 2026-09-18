# Challenge 3 — Medical Appointment (ASR + QA)

Folder: `medical-appointment/` · local port **9054** · endpoint `/predict`

## The task
One POST = one whole consultation (MP3, base64, no `data:` prefix) + **ten yes/no questions**.
No transcript given — ASR is yours. Return ten booleans, and for every `true`
the **start and end second of the passage that supports it**.

## Request / response
Request: `audio_base64`, `audio_filename` (e.g. conversation_sample_17.mp3), `questions` (10).
Response — exactly these three lists, matched **by position**, all of length = #questions:
```json
{"answers":[true,false], "evidence_start":[21.62,null], "evidence_end":[26.24,null]}
```
A wrong-length list makes the body unparseable → **all ten questions scored wrong**.
`utils.validate_response` checks it. Returning null spans for a `true` is legal (scores 0 for
evidence) and far better than a malformed body.

## Question types
| type | answer | what |
|---|---|---|
| positive | yes | something the conversation establishes |
| hard_negative | no | near-miss: same drug different dose, same symptom different place |
| off_topic | no | never comes up at all |
Yes/no are **exactly balanced** → a constant answer scores 0.500 accuracy and nothing else.
Some questions are tag questions ("…didn't it?") → **never key off sentence shape**.
Topical-similarity matching answers yes to every hard negative and lands on the floor.

## Scoring
```
Score = 0.4 * Accuracy + 0.6 * mean_tIoU
```
- Accuracy: 1 point per question, no partial credit, no confidence.
- tIoU = overlap / union of the predicted and annotated span, averaged over **all annotated
  yes questions (fixed set)** — answering `no` to a true-yes costs you on BOTH halves.
- Spans alongside a `no` are ignored (no penalty, no credit).
- **Pointing at the whole clip scores ~0.038** — return the passage, not the region.
- Baseline (all true, no spans) = 0.4*0.5 + 0 = **0.200**. That is the floor.

## Data
`data/audio/` 39 MP3s (~1–3.5 min, median ~2 min, mono 128 kbps 44.1 kHz, English, both
speakers on one channel). `data/question_train.csv` 390 rows:
`question_id, transcript_id, question, answer, label, question_type, evidence_start, evidence_end`.
Evidence columns are filled only for the **195 positive** rows; blank for the 195 no rows.

## Timing — tight
- 60 s budget per request, **and** the whole attempt gets 60 s × #conversations, so 60 s is an
  average to stay under, not an allowance to spend. Overrun → queued conversations never sent.
- Requests are strictly sequential, in file order. Nothing carries over between requests.
- **Five consecutive timeouts ends the attempt** (the tail is scored wrong). Any reply at all —
  even a 500 or a wrong-length body — resets the counter. Silence is what kills you.
- One dead request = 10 marks = >5% of a validation attempt.
- No warm-up period: load and exercise the model **at import time**.
- Budget split: if ASR takes 40 s you have ~2 s/question left for answering.

## Rules / stack
No cloud APIs inside /predict. Local ASR: `faster-whisper` (reads MP3 without ffmpeg),
`whisper.cpp`, `WhisperX` (word timings + diarization).
**On our M1 Pro, faster-whisper/CTranslate2 is CPU-only** → consider `mlx-whisper` or
`whisper.cpp` with Metal, or host on an NVIDIA box.
**Keep the segment timings** — do not join segments into one string, you need
`evidence_start`. How far to merge neighbouring segments is worth measuring against
`question_train.csv`.

Key insight from the README: *the passage you would cite to justify a yes is exactly the span
you are asked to return*, so build an evidence-finding system, not a guesser.

## Local testing
```
python api.py
python local_evaluator.py            # 390 questions
python local_evaluator.py --oracle   # must print 1.000
python local_evaluator.py --verbose
```
Read the breakdowns, not the headline: accuracy by question type, mean tIoU vs.
"tIoU when answered yes" (diagnostic, unscored), `no span returned` count, round-trip worst case.

Validation = 19 conversations / 190 questions. Evaluation = 38 / 380, **one attempt**.
