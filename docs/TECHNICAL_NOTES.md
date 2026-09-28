# Technical decisions

## Retrieval and business constraints

The Markdown loader recognizes rule IDs and keeps section headings out of adjacent rule text. The UI asks which of two business paths applies. R and B rules are routed separately; the unclear path uses G01 to ask a follow-up question. G02 is parsed as a manual entry but is not consulted by the current retrieval filter.

For vector search, the same embeddings model converts the query and manual entries into numeric vectors. Cosine similarity ranks candidates. A cache stores manual vectors based on the manual content, model name and API base URL. A rule may still be inapplicable despite a high similarity score; the implementation excludes one side of numeric 7/8-day fault boundaries and promotes two recognizable delivery or inspection-dispute intents.

The code handles Arabic numerals in day expressions. It does not generally interpret Chinese-number days such as 第九天. The intentional H09 stress case probes that limitation.

## Why the answer model sees one rule

In earlier user-run outputs, top-three context prompted the model to discuss returns and data backup while answering a payment question. The interface still shows the three candidates for inspection, but the generation call receives only the first one. This is suitable for this single-main-issue prototype and can omit secondary issues in a multi-issue question.

## Guardrails and traces

The system prompt states that the model cannot read real orders, submit tickets, connect an agent or request private data. Output checks normalize citation formatting, reject references to rule IDs outside the chosen evidence, and substitute a conservative response for some known patterns of unsupported action claims. A few obviously unrelated topics are refused before vector search.

The app records questions, retrieved rules, answer evidence, raw model output, displayed output, status and elapsed time. These traces support human review but are not anonymized. Use synthetic examples only and keep local exports out of a public repository.

## Design rationale in brief

- **Why retrieval instead of fine-tuning?** It is a small, changeable manual; retrieval exposes which rule was offered as evidence. It does not inherently ensure faithful answers.
- **Why deterministic logic as well as vectors?** Exact day boundaries and selected business intents should not depend solely on approximate similarity.
- **Is this an agent?** No. The system has no tools for order lookup, ticket submission or autonomous workflows.
- **Does 12/12 prove accuracy?** No. It measures first-rule retrieval on a small set used during tuning.
- **What would improve it?** Validate company-owned policies, build new reviewed questions, add calibrated abstention and semantic support checks, introduce consent and access controls, then integrate audited handoff tools if a real client needs them.
