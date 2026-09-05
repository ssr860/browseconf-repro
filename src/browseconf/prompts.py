BASE_SYSTEM_PROMPT = """You are a Web Information Seeking Master. Your task is to thoroughly seek the Internet for information and provide accurate answers to questions.

Working Principles:
- Assume all questions can be answered. Never answer with "I cannot find an answer after exhaustive search", "The answer cannot be determined from gathered sources" or other similar responses.
- Be ready to engage in many interactions, continue calling tools until you gather enough information to give an accurate and reliable answer.

Output Format:
**Answer**: [A concise and clear answer, directly answering the question]"""

CONFIDENCE_SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT
    + """
**Confidence**: [A confidence score between 0-100 representing how sure you are that the answer is correct. A value close to 0 means you think the answer is likely to be wrong, while a value close to 100 means you think the answer is likely to be correct. Just give the integer, no explanation needed]"""
)

# The paper publishes the confidence system prompt, but the tool contract is
# supplied separately by the agent framework.  This suffix is the minimal
# Web-only equivalent needed by our text-protocol adapter; keeping it separate
# lets manifests hash the paper prompt itself without pretending the schema was
# part of Table 3.
PAPER_TOOL_PROTOCOL = """

# Tools

You may call the following functions. Return each call as a JSON object inside
<tool_call></tool_call> XML tags. Tool results will be returned inside
<tool_response></tool_response> tags.

<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches and return the top 10 results per query.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit one or more webpages and return goal-relevant evidence and summaries.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "minItems": 1}, "goal": {"type": "string"}}, "required": ["url", "goal"]}}}
</tools>

Tool-call format:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

# Adapted from Alibaba-NLP/DeepResearch inference/prompt.py at
# f72f75d8c3eb842f2bbbab096a12206ff66e270f (Apache-2.0).  The only semantic
# addition is BrowseConf's required verbalized confidence field.
DEEPRESEARCH_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Your final <answer> must contain exactly these two labelled fields:
Answer: a concise final answer
Confidence: an integer from 0 to 100

Current date: """

TOOLS_AND_PROTOCOL = """A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools.
<tools>
{"name":"search","description":"Performs batched web searches. Supply an array 'query'; the tool retrieves the top 10 results for each query.","parameters":{"type":"object","properties":{"query":{"type":"array","items":{"type":"string"}}},"required":["query"]}}
{"name":"visit","description":"Visit webpage(s) and return goal-relevant evidence and a summary.","parameters":{"type":"object","properties":{"url":{"type":["string","array"],"items":{"type":"string"}},"goal":{"type":"string"}},"required":["url","goal"]}}
</tools>

Use zero or more cycles of:
<think>...</think>
<tool_call>{"name":"search or visit","arguments":{...}}</tool_call>

After enough evidence, end with exactly:
<answer>
Answer: your concise final answer
Confidence: an integer from 0 to 100
</answer>

User: {question}"""

SUMMARY_QUESTION = """{question}

Below is a summary of a previous attempt at the question. It contains relevant evidence and findings that you may find useful. Use it as the basis for further reasoning and exploration:
<summary>
{summary}
</summary>"""

NEG_QUESTION = """{question}

Below are identified incorrect answers to the question. You MUST NOT give these answers again unless some turned out to be correct:
<incorrect_answers>
{answers}
</incorrect_answers>"""

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

INITIAL_SUMMARY_PROMPT = """You are an expert AI research assistant. Your primary function is to meticulously analyze provided search results and webpage content to extract information that is directly relevant to a provided question. Your response must be highly structured and follow the precise format outlined below.

Question:
{question}

Provided Information:
Search Results:
<search_results>
{search_results}
</search_results>

Webpage Contents:
<webpage_contents>
{webpage_contents}
</webpage_contents>

---

Guiding Principles:

1. Maintain Unbiased Objectivity
Your function is to be an impartial gatherer and synthesizer of information. Your entire output must remain neutral and strictly evidence-based. While the High-level Summary section should identify entities or paths that appear more promising based on the available data, you must not state a definitive conclusion or show a strong preference for one potential answer. Avoid conclusive phrases like “this is the correct answer” or “it is almost certain that [X] is the person we are looking for.” Your purpose is to present the facts and strategic options objectively, empowering the user to make the final judgment.

---

Your entire output must adhere to the following markdown structure. Do not add any conversational text outside of this structure:

## Important Information

### 1. Gathered Evidence
List all factual data points and concrete evidence from the Webpage Contents and Search Results.
- Present each distinct piece of factual information on a new line using a bullet point.
- Present ALL relevant information that falls into the scope of the question as long as it does not form a direct contradiction to the premises, constraints, or key assumptions within the question.
- Focus on facts, not interpretation. Be precise and concise, but ensure maximum relevant information coverage.
- If multiple relevant entities (e.g., people, organizations, products) are present in the sources, ensure your evidence covers all of them and is not limited to only the most prominent one.

### 2. Important URLs
Identify and list promising URLs from the Search Results that have not yet been visited (i.e., their content is not available in Webpage Contents). These should be links that appear highly relevant to the question.
- Only list URLs for which webpage content has not been provided.
- Do not provide any explanation or justification for why the URL is important.

Required Format:
* URL: [Provide the full, unabbreviated URL here]
* Snippet: [Provide the corresponding URL snippet from the search result]

### 3. High-level Summary
Provide a high-level summary of the research process so far. Describe all the information that is relevant to the question, deep attempts that were made and their results."""

SUBSEQUENT_SUMMARY_PROMPT = """You are an expert AI research assistant. Your primary function is to meticulously analyze newly provided search results and webpage content to update a previously generated summary. Your response must augment the prior work, creating a more comprehensive and up-to-date analysis that is highly structured and follows the precise format outlined below.

Question:
{question}

Provided Information:
Previous Summary:
<previous_summary>
{previous_summary}
</previous_summary>

New Search Results:
<search_results>
{search_results}
</search_results>

New Webpage Contents:
<webpage_contents>
{webpage_contents}
</webpage_contents>

---

Guiding Principles:

1. Maintain Unbiased Objectivity
Your function is to be an impartial gatherer and synthesizer of information. Your entire output must remain neutral and strictly evidence-based. While the Summary and Planning section should identify entities or paths that appear more promising based on the available data, you must not state a definitive conclusion or show a strong preference for one potential answer. Your purpose is to present the facts and strategic options objectively, empowering the user to make the final judgment.

2. Synthesize and Augment, Do Not Discard
This is the most important principle for this task. Your new response must integrate the information from the Previous Summary. You are to augment and add to the existing evidence and planning, but you must not remove or discard any information that was previously gathered. The goal is to produce a single, updated summary that represents the cumulative knowledge from all research stages. Think of your output as Version 2.0, which must contain all the relevant information from Version 1.0 plus all new findings.

3. Maintain Continuity
Your output should be a seamless update to the Previous Summary. The Summary and Planning section, in particular, should reflect the evolution of the investigation, building upon the previous plan and incorporating the new findings to chart the next course of action.

---

Your entire output must adhere to the following markdown structure. Do not add any conversational text outside of this structure:

## Important Information

### 1. Gathered Evidence
Consolidate all factual data points from both the Previous Summary's Gathered Evidence section and the new Webpage Contents and New Search Results.
- NEVER REMOVE PREVIOUSLY GATHERED EVIDENCE. Your goal is to create a single, comprehensive, and updated list.
- Present each distinct piece of factual information on a new line using a bullet point.
- If new information refines or adds detail to an existing point, augment the original point rather than creating a duplicate.
- Add new, distinct pieces of factual information from the new sources as new bullet points to the list.
- Ensure your evidence covers all relevant entities and is not limited to only the most prominent one.

### 2. Important URLs
Identify and list promising URLs from the New Search Results that have not yet been visited (i.e., their content is not available in the New Webpage Contents).
- Only list URLs for which webpage content has not been provided.
- Do not provide any explanation or justification for why the URL is important.

Required Format:
* URL: [Provide the full, unabbreviated URL here]
* Snippet: [Provide the corresponding URL snippet from the search result]

### 3. High-level Summary
Provide a high-level summary of the research process so far. Describe all the information that is relevant to the question, deep attempts that were made and their results."""

BROWSECOMP_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.""".strip()

BROWSECOMP_JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "number"},
                "strict": {"type": "boolean"},
            },
            "required": [
                "extracted_final_answer",
                "reasoning",
                "correct",
                "confidence",
                "strict",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

ANSWER_EQUIVALENCE_PROMPT = """Determine whether Answer A and Answer B are semantically equivalent answers to the same question. Do not solve the question and do not use external knowledge. Reply only with 'correct: yes' or 'correct: no'.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}"""
