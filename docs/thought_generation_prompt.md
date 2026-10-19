# Writing rows of the art thought dataset

The dataset: AbstractPhil/thought-symbolic-image-captions. The plan: history/plans/2026-10-19_art_thought_dataset_plan.md in the research record. The tooling:
`geolip.nano_beatrix.thought_pack` (this document is written by it; the tests keep them equal).

A row is one act of THINKING ABOUT AN IMAGE that nobody sees: a hidden image specification in the systems trie, a chain of thoughts
at a stated depth, the wordings an artist would use, and questions whose answers sit in the records. Text only; nothing refers to
pixels or to seeing a file.

## The depths
| depth | the thought holds | records | thoughts | questions | wordings | frame |
|---|---|---|---|---|---|---|
| simple | one observation or one naming | 1-4 | 1-2 | 1-2 | 1 | 128-256 bytes |
| medium | subjects with attributes, their relations, the form, a definition, the behavior shown | 6-14 | 3-5 | 3-5 | 1 | 512 bytes |
| deep | the whole composition, mood and intent with evidence, a comparison or a change, the story, a critique | 14-30 | 6-10 | 5-8 (one unknown, one needing the chain) | 3 (caption, prompt, tags) | 1,024-2,048 bytes |

Rows come in SERIES of three on one hidden image: ranks 1, 2, 3 = simple, medium, deep, each spec a subset of the next.

## The rules
1. every claim is in the spec or is marked as an inference with its evidence; nothing refers to pixels or to seeing the file
2. record values are plain lower-case words of at most 20 bytes, keys from the trie; longer language lives in the thoughts and the wordings
3. the depth sets the counts of records, thoughts and questions
4. every question's value is a record's value, 'unknown' (a key the spec does not hold), or (needs 'chain') a value the thoughts state
5. at least one wording a row, three for deep (caption, prompt, tags); the tag wording uses the booru register where it exists
6. a mood names its evidence: the key of the record that shows it (look expression, shot light, rel1 rel)
7. a series shares its spec by subset: simple within medium within deep
8. no real person's name, no copyrighted image described from memory: the spec is invented or comes from a licensed description

## The systems trie
- person: count, gender, age, role
- look: hair, eyes, expression, makeup, jewelry, skin
- wear: clothing, accessory, colour
- body: pose, form, gesture
- scene: setting, time, weather, layout, mood
- shot: light, angle, framing, lens, focus
- thing: name, count, colour, state, position
- form: composition, balance, line, shape, space, rhythm, contrast, palette, texture, medium, style, period
- mood: feeling, tone, energy, temperature, evidence
- intent: purpose, audience, message, genre
- story: moment, before, after, cause, teller
- relation: a list of {a, rel, b} with rel from: left of, right of, above, below, behind, in front of, beside, holding, facing, looking at, larger than, smaller than, same colour as, touching, inside, under, leaning on, reflected in

## The schema (an example row of depth medium)
```json
{
 "id": "art-000123",
 "series": "s-00041",
 "depth": "medium",
 "rank": 2,
 "domain": "photograph",
 "genre": "street portrait",
 "spec": {
  "person": {
   "gender": "woman"
  },
  "look": {
   "hair": "short",
   "expression": "wry"
  },
  "wear": {
   "clothing": "raincoat",
   "colour": "yellow"
  },
  "scene": {
   "setting": "market street",
   "time": "dusk",
   "weather": "rain"
  },
  "shot": {
   "light": "neon",
   "framing": "waist up"
  },
  "form": {
   "composition": "subject right third"
  },
  "relation": [
   {
    "a": "woman",
    "rel": "in front of",
    "b": "fish stall"
   }
  ],
  "mood": {
   "feeling": "amused",
   "evidence": "look expression"
  }
 },
 "thoughts": [
  {
   "step": "see",
   "text": "A woman in a yellow raincoat, short hair, a wry half smile."
  },
  {
   "step": "place",
   "text": "She stands on a market street at dusk, in the rain, in front of a fish stall."
  },
  {
   "step": "form",
   "text": "Neon lights her face; she sits in the right third of the frame, seen from the waist up."
  },
  {
   "step": "mean",
   "text": "The wry expression reads as amused rather than weary. With dusk and neon the frame is dark around a single subject: an inference from the time and the light, not a record."
  },
  {
   "step": "say",
   "text": "The composed wording follows."
  }
 ],
 "wordings": {
  "caption": "A woman with short hair and a wry smile, in a yellow raincoat, on a rainy market street at dusk, lit by neon, in front of a fish stall.",
  "prompt": "street portrait, woman, short hair, wry smile, yellow raincoat, rainy market street at dusk, neon light, fish stall behind, waist up, subject right third",
  "tags": "1girl, short hair, smile, raincoat, rain, dusk, neon lights, market, portrait, upper body"
 },
 "questions": [
  {
   "key": "look hair",
   "value": "short",
   "needs": "record"
  },
  {
   "key": "rel1 b",
   "value": "fish stall",
   "needs": "record"
  },
  {
   "key": "scene layout",
   "value": "single subject",
   "needs": "chain"
  },
  {
   "key": "wear accessory",
   "value": "unknown",
   "needs": "unknown"
  }
 ],
 "held": {
  "layout": null,
  "wordings": false,
  "values": []
 },
 "source": {
  "kind": "synthetic",
  "model": "sonnet",
  "seed_dataset": null,
  "license": "CC BY 4.0"
 }
}
```

## The protocol
1. The vocabulary file and the sealed held-out values first (`python -m geolip.nano_beatrix.thought_pack --vocab=<folder>`): a tenth of
   every closed list is sealed; training rows never use a sealed value, the held rows for the reads use them.
2. The spec before the thought: draw or receive the hidden spec, then the thoughts, then the wordings, then the questions.
3. A series is written together so the subsets hold.
4. The diversity knobs are set per batch and logged: the depth mix, the domain, the genres, the register, the share of unknown
   questions, the share of negative facts ("accessory: none"), relations in every medium and deep row.
5. Every row passes the validator (`--check=<rows.jsonl>`); the batch log keeps the problems by rule.
6. A pilot of 2,000 rows comes before volume.

## The prompt
`python -m geolip.nano_beatrix.thought_pack --prompt=1 --depth=medium --domain=photograph` prints the instruction for one call (the
schema, the rules, the lists the model may draw from); `--series=1` asks for series of three; `--split=held` writes rows on the sealed
values for the reads. `--sample=N --series=1 --seed=S` draws N nested spec triples (the facts) for the model to think from, so the lists
are covered evenly; the model adds mood (with its evidence), intent and story.
