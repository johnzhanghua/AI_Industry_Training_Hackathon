# Evaluation results -- `domain-ft`

- Run (UTC): 2026-07-31T05:01:58+00:00
- Endpoint: http://localhost:4000
- Questions: ../../Participant_Package/public_questions.jsonl (15 public questions)
- Judge: agent-brain (component-based rubric)

## Similarity metrics (BLEU / ROUGE / F1)

```
        ungrounded  grounded   delta
bleu        0.0388    0.1575  0.1187
rouge1      0.2475    0.4675  0.2200
rougeL      0.1871    0.3518  0.1647
f1          0.1991    0.3713  0.1723
```

## Component-based LLM judge (normalized, 0-1)

```
condition
grounded      0.639
ungrounded    0.107
```

Grounding delta: +0.532

### By difficulty

```
condition   grounded  ungrounded  delta
difficulty                             
easy           0.750       0.200  0.550
hard           0.729       0.100  0.629
medium         0.524       0.057  0.467
```

### Per question

```
    id difficulty  condition  score  normalized
MHQ035     medium ungrounded   0.00       0.000
MHQ090       hard ungrounded   0.00       0.000
MHQ049     medium ungrounded   0.00       0.000
MHQ074       hard ungrounded   0.00       0.000
MHQ055       hard ungrounded   0.00       0.000
MHQ001       easy ungrounded   0.00       0.000
MHQ040       easy ungrounded   0.00       0.000
MHQ076       easy ungrounded   0.00       0.000
MHQ045     medium ungrounded   0.00       0.000
MHQ061     medium ungrounded   0.00       0.000
MHQ072     medium ungrounded   0.00       0.000
MHQ084     medium ungrounded   0.00       0.000
MHQ067       hard ungrounded   4.00       0.400
MHQ058       easy ungrounded   8.00       0.800
MHQ080     medium ungrounded   4.00       0.400
MHQ061     medium   grounded   0.00       0.000
MHQ001       easy   grounded  10.00       1.000
MHQ035     medium   grounded   0.00       0.000
MHQ049     medium   grounded   0.00       0.000
MHQ090       hard   grounded   6.67       0.667
MHQ058       easy   grounded  10.00       1.000
MHQ045     medium   grounded  10.00       1.000
MHQ040       easy   grounded  10.00       1.000
MHQ072     medium   grounded   6.66       0.666
MHQ067       hard   grounded  10.00       1.000
MHQ080     medium   grounded  10.00       1.000
MHQ074       hard   grounded   2.50       0.250
MHQ055       hard   grounded  10.00       1.000
MHQ076       easy   grounded   0.00       0.000
MHQ084     medium   grounded  10.00       1.000
```
