"""LLM-as-judge for generated posts. Forces structured output via tool-use."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import anthropic

JUDGE_TOOL = {
    "name": "record_judgment",
    "description": (
        "Record your judgment of a generated social media post against a rubric. "
        "Score each criterion strictly and provide concise notes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "voice_match": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How closely the post matches the voice of the provided Capital Group examples. 1=very off-voice. 5=indistinguishable from a real Capital Group post.",
            },
            "on_topic": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How accurately the post represents the article's actual content. Penalize hallucinated facts/stats. 1=disconnected or hallucinated. 5=tight and accurate.",
            },
            "hook_strength": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How compelling the opening hook is. 1=generic/limp. 5=instantly engaging.",
            },
            "compliance": {
                "type": "string",
                "enum": ["pass", "fail"],
                "description": (
                    "Pass if the post avoids ALL of: specific return claims (e.g. 'returns 15%'), "
                    "buy/sell verbs ('buy now', 'sell X'), guarantee language ('guaranteed', "
                    "'will return'), and single-ticker mentions (e.g. 'AAPL', 'NYSE:JPM'). "
                    "Fail otherwise."
                ),
            },
            "compliance_notes": {
                "type": "string",
                "description": "If compliance='fail', specify the exact phrase that triggered failure. Otherwise empty string.",
            },
            "overall_notes": {
                "type": "string",
                "description": "One or two sentences summarizing strengths and weaknesses.",
            },
        },
        "required": [
            "voice_match",
            "on_topic",
            "hook_strength",
            "compliance",
            "compliance_notes",
            "overall_notes",
        ],
    },
}

JUDGE_SYSTEM_TEMPLATE = """You are a strict editorial reviewer for Capital Group's marketing team.

Your job is to evaluate a generated social media post against a rubric. Capital Group is a regulated US asset manager, so compliance must be evaluated literally and strictly.

Compliance fails if the post contains any of:
- A specific investment return claim (e.g. "returns 15%", "up 28% in 6 months")
- A buy/sell verb directed at investors ("buy X now", "sell Y")
- Guarantee language ("guaranteed return", "will deliver", "promised yield")
- A single-ticker mention (e.g. "AAPL", "NYSE:JPM", "$TSLA")

Compliance does NOT fail for:
- General references to market performance ("stocks rose")
- Historical data drawn from the article ("the S&P fell 28.5% over three months")
- Mentions of asset classes (bonds, equities, cash)
- Hedge language ("could", "may", "potentially")

Use the record_judgment tool to record your judgment. Do not respond with prose."""


CALIBRATION_BLOCK_TEMPLATE = """Here are examples of how to judge posts:

{examples_text}

Now judge the following post."""


@dataclass
class JudgeResult:
    voice_match: int
    on_topic: int
    hook_strength: int
    compliance: Literal["pass", "fail"]
    compliance_notes: str
    overall_notes: str
    raw_response: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    @property
    def soft_score_sum(self) -> int:
        return self.voice_match + self.on_topic + self.hook_strength


@dataclass
class LengthCheck:
    char_count: int
    in_envelope: bool
    target_min: int
    target_max: int
    feedback: str


def check_length(post: str, target_min: int, target_max: int) -> LengthCheck:
    """Pure Python length check. Returns retry feedback string if out of envelope."""
    n = len(post)
    if n < target_min:
        feedback = (
            f"Your previous post was {n} chars, below the target minimum of {target_min}. "
            f"Expand it to {target_min}-{target_max} chars."
        )
        return LengthCheck(n, False, target_min, target_max, feedback)
    if n > target_max:
        feedback = (
            f"Your previous post was {n} chars, above the target maximum of {target_max}. "
            f"Tighten it to {target_min}-{target_max} chars without losing core content."
        )
        return LengthCheck(n, False, target_min, target_max, feedback)
    return LengthCheck(n, True, target_min, target_max, "")


def format_calibration_examples(examples: list[dict]) -> str:
    """Render the few-shot calibration examples from config into prompt text."""
    parts = []
    for i, ex in enumerate(examples, start=1):
        parts.append(
            f"--- Example {i} ---\n"
            f"Post: {ex['post']}\n"
            f"Judgment:\n"
            f"  voice_match: {ex['voice_match']}\n"
            f"  on_topic: {ex['on_topic']}\n"
            f"  hook_strength: {ex['hook_strength']}\n"
            f"  compliance: {ex['compliance']}\n"
            f"  compliance_notes: {ex.get('compliance_notes', '')}\n"
            f"  reasoning: {ex['reasoning']}"
        )
    return "\n\n".join(parts)


def build_judge_user_blocks(
    article: dict,
    voice_examples: list[dict],
    calibration_examples: list[dict],
    post_to_judge: str,
) -> list[dict]:
    """Build judge user content. Cache breakpoint on the article + examples block."""
    voice_text = "\n\n".join(
        f"<example>\n{ex['text_clean']}\n</example>" for ex in voice_examples
    )
    calibration_text = format_calibration_examples(calibration_examples)

    cached_text = f"""{CALIBRATION_BLOCK_TEMPLATE.format(examples_text=calibration_text)}

For context, here are real Capital Group posts in the relevant voice (audience track: {article['audience_track']}):

{voice_text}

The post is promoting this article:
Title: {article.get('title', '')}
URL: {article['url']}

Article body (for grounding the on_topic check):
{(article.get('body') or article.get('title', ''))[:3000]}"""

    delta_text = f"""Post to judge:
\"\"\"
{post_to_judge}
\"\"\"

Call the record_judgment tool with your judgment."""

    return [
        {"type": "text", "text": cached_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": delta_text},
    ]


def judge_post(
    client: anthropic.Anthropic,
    model: str,
    article: dict,
    voice_examples: list[dict],
    calibration_examples: list[dict],
    post: str,
) -> JudgeResult:
    """Single judge call. Forces tool-use for structured output."""
    user_blocks = build_judge_user_blocks(article, voice_examples, calibration_examples, post)

    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=[
            {
                "type": "text",
                "text": JUDGE_SYSTEM_TEMPLATE,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "record_judgment"},
        messages=[{"role": "user", "content": user_blocks}],
    )

    tool_call = next(b for b in response.content if b.type == "tool_use")
    j = tool_call.input

    return JudgeResult(
        voice_match=j["voice_match"],
        on_topic=j["on_topic"],
        hook_strength=j["hook_strength"],
        compliance=j["compliance"],
        compliance_notes=j["compliance_notes"],
        overall_notes=j["overall_notes"],
        raw_response=j,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        },
    )
