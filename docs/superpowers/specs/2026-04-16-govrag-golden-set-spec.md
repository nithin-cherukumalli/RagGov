# GovRAG Golden Set Spec

Date: 2026-04-16

## Goal

Define a human-reviewed golden evaluation set derived from the actual corpus in `tests/Data` that does two jobs at the same time:

1. measure answer quality in the way teams expect from frameworks such as RAGAS or DeepEval
2. tell the developer what failed, where it failed, and how to fix it

This is the central difference between generic RAG evaluation and GovRAG. The golden set must not stop at “bad answer” or “low score.” It must support failure attribution across parsing, chunking, embedding, retrieval, grounding, sufficiency, security, and confidence.

## Scope Decision

The first pass should **not** attempt full coverage over every PDF in `tests/Data`.

That would give breadth but poor diagnostic quality. A better first version is a **high-signal subset** covering the distinct document behaviors already present in the corpus:

- deeply nested rules
- multi-page tables and annexures
- near-duplicate personnel orders
- short procedural orders with legal references
- budget and policy orders with enumerated permissions

This gives GovRAG a realistic chance to both answer questions and explain failures.

## Selected Document Subset

### A. Deep hierarchy and cross-section reasoning

- `2011SE_MS20.PDF`

Why it matters:
- 39 pages
- true rules document
- nested sections, sub-rules, definitions, operative provisions
- best source for hierarchy-sensitive and sufficiency-sensitive evaluation

### B. Table-heavy and structured allocation orders

- `2011SE_MS9.PDF`
- `2011SE_MS15.PDF`
- `2011SE_MS39.PDF`

Why they matter:
- annexures / statements
- numeric allocations
- district/category mappings
- row-column semantics that break easily under weak parsing and chunking

### C. Near-duplicate retrieval confusion cluster

- `2011SE_MS24.PDF`
- `2011SE_MS29.PDF`
- `2011SE_MS30.PDF`
- `2011SE_MS35.PDF`

Why they matter:
- almost identical wording
- different names, districts, and personnel
- ideal for embedding collapse, ranking mistakes, and citation drift

### D. Medium narrative orders with policy details

- `2011SE_MS1.PDF`
- `2011SE_MS11.PDF`
- `2011SE_MS16.PDF`
- `2011SE_MS17.PDF`
- `2011SE_MS22.PDF`

Why they matter:
- compact but semantically meaningful
- mix of legal authority, financial approval, personnel action, and factual details
- useful for direct QA, multi-clause QA, and abstention tests

## Evaluation Philosophy

Every gold item must support two parallel judgments.

### 1. Answer-quality judgment

For each question:
- is the answer correct
- is it complete enough
- is it grounded in the corpus
- are the cited/supporting sections right
- should the system have abstained instead of answering

### 2. Failure-attribution judgment

For the same question:
- if the answer is wrong, what stage most likely failed first
- what exact condition caused the failure
- whether the failure is deterministic or query-dependent
- what remediation should the developer try first

This means each gold item is not just `question -> answer`. It is:

- question
- gold answer
- support locations
- expected abstention behavior
- likely failure signatures
- diagnostic hints

## Golden Set Schema

Each item in the golden set should eventually materialize to a machine-readable record with these fields:

- `gold_id`
- `source_docs`
- `question`
- `question_type`
- `difficulty`
- `gold_answer`
- `gold_short_answer`
- `supporting_evidence`
- `required_doc_ids`
- `required_sections`
- `should_abstain_if_missing_support`
- `answer_quality_checks`
- `diagnostic_expectations`
- `likely_primary_failure_if_wrong`
- `likely_secondary_failures_if_wrong`
- `developer_fix_hint`

### Supporting Evidence Format

Each evidence entry should include, when possible:

- `doc_id`
- `page`
- `section_label`
- `quoted_fact_summary`

We are not optimizing for long quotes. We are optimizing for precise anchoring.

## Question Types to Include

The first golden set should cover these categories:

1. direct factual lookup
2. multi-clause synthesis within one document
3. cross-section reasoning within one document
4. table lookup
5. table + narrative synthesis
6. near-duplicate disambiguation
7. legal authority / basis identification
8. date / amount / count extraction
9. amendment / deletion / exception handling
10. abstention-required questions

## Gold Set Composition

Recommended first version: **48 questions**

- 10 from `MS20`
- 10 across `MS9`, `MS15`, `MS39`
- 12 across the near-duplicate cluster `MS24`, `MS29`, `MS30`, `MS35`
- 16 across `MS1`, `MS11`, `MS16`, `MS17`, `MS22`

This is enough to expose failure patterns without becoming too large to review carefully.

## Document-by-Document Gold Design

## 1. `MS20` Golden Questions

Document role:
- primary hierarchy benchmark
- sufficiency benchmark
- cross-section reasoning benchmark

### Gold questions

1. What is the neighborhood distance for classes I-V?
   - type: direct factual
   - gold answer: one kilometer walking distance
   - support: Rule 5(1)(a)

2. What is the neighborhood distance for classes VI-VIII?
   - type: direct factual
   - gold answer: three kilometers walking distance
   - support: Rule 5(1)(b)

3. What should the government do where no school exists within the neighborhood limits?
   - type: cross-section reasoning
   - gold answer: make arrangements such as free transportation, residential facilities, and other facilities
   - support: Rule 5(4)

4. What is the minimum duration of special training and what is the maximum extension?
   - type: multi-clause factual
   - gold answer: minimum three months, extendable up to two years
   - support: Rule 4(1)(d)

5. Who may be involved in mobilization and identification of out-of-school children?
   - type: direct factual
   - gold answer: civil society organizations and self help groups
   - support: Rule 4(3)

6. What does the rule say about children with disabilities who cannot access school?
   - type: cross-section reasoning
   - gold answer: transport should be arranged where possible; severe disability may require home-based education
   - support: Rule 5(7)

7. What does “free education” include under these rules?
   - type: definition extraction
   - gold answer: no direct or indirect costs; includes textbooks, notebooks, writing material, midday meals, uniforms in neighborhood government schools
   - support: Definitions clause (13)

8. What is meant by “out of school child”?
   - type: definition extraction
   - gold answer: age 6-14 child who has not completed elementary education, never enrolled, dropped out, or absent for more than one month
   - support: Definitions clause (19)

9. Which authority identifies neighborhood schools and makes the information public?
   - type: factual lookup
   - gold answer: the local authority
   - support: Rule 5(6)

10. Abstention test: What 2012 district budget was allocated to enforce Rule 5?
   - type: abstention
   - gold answer: abstain / not in provided corpus
   - support: none

### Diagnostic use

If these fail:
- hierarchy loss -> parsing/chunking
- clause missing from retrieval -> retrieval
- unsupported synthesis -> grounding
- invented budget -> sufficiency / grounding / confidence

## 2. `MS9`, `MS15`, `MS39` Golden Questions

These are the table-sensitive questions.

### `MS9`

1. How many posts can be adjusted to SSA for Krishna district?
   - gold answer: 228
   - support: annexure row for Krishna, “Posts that can be adjusted to SSA”

2. What is the total number of teacher posts adjusted to SSA?
   - gold answer: 22075
   - support: paragraph 5 and annexure total

3. How many posts were allotted to M.A. & U.D. Department?
   - gold answer: 1027
   - support: paragraph 3

4. What was the total created in G.O.Ms.No.363?
   - gold answer: 26859
   - support: annexure total

### `MS15`

5. How many categories were identified for rectification of pay anomalies?
   - gold answer: 19
   - support: paragraph 3

6. Which categories had financial implication?
   - gold answer: serial numbers 1, 2, 3, 5, 6, 7, and 8
   - support: paragraph 4

7. What monetary period and amount were worked out?
   - gold answer: from 1-4-2005; Rs.11,57,724
   - support: paragraph 4

8. From what dates were the revised scales implemented notionally and monetarily?
   - gold answer: notionally from 01-07-2003 and monetary benefit from 01-04-2005
   - support: paragraph 5

### `MS39`

9. What is the proposed pattern of Project Officers for Warangal?
   - gold answer: 1
   - support: Statement-A

10. What is the proposed pattern of Project Officers for Khammam?
   - gold answer: 2
   - support: Statement-A

11. What condition applies to APOs and Supervisors regarding transfers?
   - gold answer: transfers shall be within the zone only
   - support: paragraph 3(d)

12. How many total APO and Supervisor posts are proposed?
   - gold answer: 47 APO and 254 Supervisors
   - support: Statement-B totals

### Diagnostic use

If these fail:
- row-value mismatch -> parsing / embedding / chunking
- totals wrong but doc retrieved -> grounding
- district confusion -> retrieval
- condition omitted -> chunking or sufficiency

## 3. Near-Duplicate Cluster Golden Questions

Documents:
- `MS24`
- `MS29`
- `MS30`
- `MS35`

These questions are not hard because of reasoning. They are hard because of disambiguation.

### Gold questions

1. Which retired employee in Medak District had her repatriation deletion order issued?
   - from `MS22`
   - gold answer: Smt P. Mary Usha Rani
   - included here as a contrast disambiguation case

2. Which retired teacher in Mahabubnagar District faced proceedings under Rule 9?
   - gold answer: Sri Shanker Reddy
   - support: `MS35`

3. Which retired teacher in Nalgonda District from Utlalpalli, Miryalguda faced proceedings?
   - gold answer: Sri K.V.L. Narsimha Rao
   - support: `MS30`

4. Which retired S.A. from Ikkurthy, Alair, Nalgonda District was proceeded against?
   - gold answer: Sri P. Rajaiah
   - support: `MS29`

5. Which retired SGT from Wardhannapet, Warangal District was proceeded against?
   - gold answer: Sri B. Sambaiah
   - support: `MS24`

6. Under which rule were departmental proceedings sanctioned in these vigilance orders?
   - gold answer: Rule 9 of the A.P. Revised Pension Rules, 1980
   - support: all duplicate cluster docs

7. What further rule governs how the proceedings shall be conducted?
   - gold answer: Rule 20 of A.P.C.S. (CC&A) Rules, 1991
   - support: all duplicate cluster docs

8. What allegation appears in these vigilance orders?
   - gold answer: fake, forged, fabricated medical bills for reimbursement
   - support: all duplicate cluster docs

### Diagnostic use

If these fail:
- wrong name, right district pattern -> retrieval ranking
- right answer with wrong source -> citation mismatch
- mixed identity attributes -> grounding / retrieval
- all same answer across docs -> embedding collapse

## 4. Narrative Policy / Order Questions

### `MS1`

1. What special relief was granted to Smt. T. Jyothi?
   - gold answer: reinstatement into service as Secondary Grade Teacher as a special case

2. How was the out-of-employment period to be treated?
   - gold answer: dies-non

3. Why had the earlier request been rejected?
   - gold answer: she contested as a candidate while holding office/place of profit related restrictions applied

### `MS11`

4. Why were the four teachers repatriated?
   - gold answer: they were appointed in violation of Presidential Order 1975 under local reservation rules

5. Under which provision did the government order repatriation?
   - gold answer: para 5(2)(c) of Presidential Order, 1975

### `MS16`

6. For which classes was syllabus review permitted?
   - gold answer: classes VII to X

7. What additional budget was requested for syllabus review and textbook preparation?
   - gold answer: Rs.1,94,51,595 requested; Finance issued Rs.1,94,52,000

8. What moral education text was proposed?
   - gold answer: Life and Teachings of Swami Vivekananda

### `MS17`

9. To how many adhoc teachers was RPS 2010 extended?
   - gold answer: 21

10. Was the extension retrospective or prospective?
   - gold answer: prospective, from the date of issue of orders

11. Did extension of minimum time scale confer regularization rights?
   - gold answer: no

### `MS22`

12. Whose name was deleted from the repatriation list in G.O.Ms.No.158?
   - gold answer: Smt P. Mary Usha Rani

13. Why was her name deleted?
   - gold answer: recalculation under the notified posts meant she came under 20% open competition and could be retained in Medak

## Scoring Dimensions

Every gold item should eventually produce these evaluation dimensions:

### Standard answer-quality dimensions

- correctness
- completeness
- grounding
- citation accuracy
- abstention correctness

### GovRAG-specific dimensions

- failure origin accuracy
- failure condition specificity
- developer remediation usefulness
- recoverability classification

## Failure Attribution Mapping

When a question fails, the golden set should guide expected attribution like this:

- wrong metadata / section / row extracted -> Parsing
- evidence split across chunks -> Chunking
- semantically similar docs collapsed -> Embedding
- wrong doc ranked top-k -> Retrieval
- answer unsupported despite evidence being present -> Grounding
- answer attempted despite absent evidence -> Sufficiency
- unsafe instruction-bearing chunk influences answer -> Security
- low-signal or conflicting pipeline state without clear hard failure -> Confidence

## Minimal First Gold Pack

If we want the most practical first deliverable, the first materialized pack should include:

- 10 `MS20` questions
- 12 table/annexure questions across `MS9`, `MS15`, `MS39`
- 8 near-duplicate retrieval questions across `MS24`, `MS29`, `MS30`, `MS35`
- 10 narrative/policy questions across `MS1`, `MS11`, `MS16`, `MS17`, `MS22`
- 8 abstention / negative-control questions spread across the same docs

Total: **48**

## Recommendation

Proceed in two steps:

1. Review and approve this gold-spec structure
2. Materialize the first 48-item machine-readable golden set under `stresslab/cases/golden/` with:
   - exact questions
   - exact gold answers
   - exact support anchors
   - expected failure-attribution hints

This is the right path for GovRAG. It lets the project compete with scoring-only frameworks while doing the thing those frameworks usually cannot do: tell the developer what went wrong and how to fix it.
