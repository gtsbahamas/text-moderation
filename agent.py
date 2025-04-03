from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logfire
from httpx import AsyncClient
from pydantic_ai import Agent, RunContext

from agent_tools import (
    check_text_toxicity,
    detect_sensitive_information,
    categorize_content,
    suggest_content_edits,
    get_moderation_decision,
)

from agent_prompts import SYSTEM_PROMPT

# Configure logging
logfire.configure(send_to_logfire='if-token-present')

@dataclass
class Deps:
    """Dependencies for the Text Moderation Agent."""
    client: AsyncClient
    moderation_api_key: Optional[str] = None
    custom_content_policies: Optional[Dict[str, List[str]]] = None
    sensitivity_threshold: float = 0.7
    store_flagged_content: bool = True
    auto_moderate: bool = False

text_moderation_agent = Agent(
    'openai:gpt-4o',
    system_prompt=SYSTEM_PROMPT,
    deps_type=Deps,
    retries=2,
)

@text_moderation_agent.tool
async def analyze_text(
    ctx: RunContext[Deps], 
    text: str,
    context: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze text for content moderation purposes.

    Args:
        ctx: The run context containing dependencies.
        text: The text content to analyze.
        context: Optional context about where the text appears.
        user_id: Optional identifier for the user who created the content.

    Returns:
        A dictionary containing the analysis results.
    """
    with logfire.span('analyzing_text', 
                     attributes={'text_length': len(text), 'has_context': context is not None}):

        # Get toxicity scores
        toxicity_result = await check_text_toxicity(ctx, text)

        # Detect sensitive information
        sensitive_info = await detect_sensitive_information(ctx, text)

        # Categorize content 
        categories = await categorize_content(ctx, text, context)

        return {
            "text": text,
            "toxicity": toxicity_result,
            "sensitive_information": sensitive_info,
            "categories": categories,
            "context": context,
            "user_id": user_id,
        }

@text_moderation_agent.tool
async def get_moderation_recommendation(
    ctx: RunContext[Deps],
    analysis_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Get a moderation recommendation based on text analysis.

    Args:
        ctx: The run context containing dependencies.
        analysis_result: The result from the analyze_text tool.

    Returns:
        A dictionary with the moderation decision and explanation.
    """
    with logfire.span('getting_moderation_recommendation'):

        decision = await get_moderation_decision(ctx, analysis_result)

        # If the content is flagged and auto-moderate is enabled, generate suggested edits
        suggested_edits = None
        if decision["action"] != "approve" and ctx.deps.auto_moderate:
            suggested_edits = await suggest_content_edits(ctx, analysis_result["text"], decision["reasons"])

        return {
            "original_text": analysis_result["text"],
            "decision": decision["action"],  # "approve", "flag_for_review", or "reject"
            "confidence": decision["confidence"],
            "reasons": decision["reasons"],
            "policy_violations": decision["policy_violations"],
            "suggested_edits": suggested_edits,
            "toxicity_scores": analysis_result["toxicity"],
            "categories": analysis_result["categories"],
        }

async def main():
    """
