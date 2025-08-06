import os
import re
import traceback
from typing import Annotated
from textwrap import dedent

from beeai_framework.adapters.openai import OpenAIChatModel
from dotenv import load_dotenv

from a2a.types import AgentCapabilities, AgentSkill, Message
from beeai_sdk.server import Server
from beeai_sdk.server.context import Context
from beeai_sdk.a2a.extensions import AgentDetail, AgentDetailTool
from beeai_sdk.a2a.extensions.ui.citation import CitationExtensionServer, CitationExtensionSpec
from beeai_sdk.a2a.extensions.ui.trajectory import TrajectoryExtensionServer, TrajectoryExtensionSpec

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.message import UserMessage, AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool

load_dotenv()

server = Server()
memories = {}

def get_memory(context: Context) -> UnconstrainedMemory:
    """Get or create session memory"""
    
    context_id = getattr(context, "context_id", getattr(context, "session_id", "default"))
    return memories.setdefault(context_id, UnconstrainedMemory())

def extract_citations(text: str, search_results=None) -> tuple[list[dict], str]:
    """Extract citations and clean text - returns citations in the correct format"""
    citations, offset = [], 0
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    
    for match in re.finditer(pattern, text):
        content, url = match.groups()
        start = match.start() - offset

        citations.append({
            "url": url,
            "title": url.split("/")[-1].replace("-", " ").title() or content[:50],
            "description": content[:100] + ("..." if len(content) > 100 else ""),
            "start_index": start, 
            "end_index": start + len(content)
        })
        offset += len(match.group(0)) - len(content)

    return citations, re.sub(pattern, r"\1", text)

def is_casual(msg: str) -> bool:
    """Check if message is casual/greeting"""
    casual_words = {'hey', 'hi', 'hello', 'thanks', 'bye', 'cool', 'nice', 'ok', 'yes', 'no'}
    words = msg.lower().strip().split()
    return len(words) <= 3 and any(w in casual_words for w in words)

@server.agent(
    name="Jenna's Granite Chat",
    detail=AgentDetail(
        ui_type="chat",
        user_greeting="Hi! I'm your Granite-powered AI assistant—here to help with questions, research, and more. What can I do for you today?",
        tools=[
            AgentDetailTool(
                name="Think", 
                description="Advanced reasoning and analysis to provide thoughtful, well-structured responses to complex questions and topics."
            ),
            AgentDetailTool(
                name="DuckDuckGo", 
                description="Search the web for current information, news, and real-time updates on any topic."
            )
        ],
        framework="BeeAI",
        author={
            "name": "Jenna Winkler"
        },
        source_code_url="https://github.com/jenna-winkler/granite_chat"
    ),
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id="chat",
            name="Chat",
            description=dedent(
                """\
                The agent is an AI-powered conversational system designed to process user messages, maintain context,
                and generate intelligent responses.
                """
            ),
            tags=["Chat"],
            examples=[
                "What are the latest advancements in AI research from 2025?",
                "Summarize the key points from the OpenAI Dev Day 2024 announcement.",
                "How does quantum computing differ from classical computing? Explain like I'm in high school.",
                "What's the difference between LLM tool use and API orchestration?",
                "Can you help me draft an email apologizing for missing a meeting?",
            ]

        )
    ],
)
async def general_chat_assistant(
    input: Message, 
    context: Context,
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()]
):
    """
    This is a general-purpose chat assistant prototype built with the BeeAI Framework and powered by Granite. 
    It leverages the experimental `RequirementAgent` with `ConditionalRequirement` rules to intelligently decide 
    when to use tools—specifically `ThinkTool` for reasoning and `DuckDuckGoSearchTool` for fetching real-time information.

    The implementation uses specific conditional requirements: `ThinkTool` is forced at step 1 and after any other 
    tool execution (with `consecutive_allowed=False`), while `DuckDuckGoSearchTool` is limited to 2 invocations maximum 
    and includes custom checks that skip search for casual messages like "hi" or "thanks."

    It maintains conversation context using `UnconstrainedMemory` with session-based storage and implements comprehensive 
    trajectory metadata logging throughout all interaction steps. Search results are automatically processed through 
    regex-based citation extraction that converts markdown links `[text](URL)` into structured `CitationMetadata` objects 
    for the platform's citation GUI support.

    The agent includes error handling with try-catch blocks that provide clear, helpful messages when issues occur, and 
    uses the `is_casual()` function to intelligently determine when tools aren't necessary for simple conversational exchanges.
    """

    user_msg = input.parts[0].root.text if input.parts else "Hello"
    memory = get_memory(context)
    
    yield trajectory.trajectory_metadata(
        title="Processing Message",
        content=f"💬 Processing: '{user_msg}'"
    )
    
    try:
        await memory.add(UserMessage(user_msg))
        
        OpenAIChatModel.tool_choice_support = set()
        llm = OpenAIChatModel(
            model_id=os.getenv('LLM_MODEL', 'llama3.1'),
            base_url=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "dummy")
        )
        
        agent = RequirementAgent(
            llm=llm, memory=memory,
            tools=[ThinkTool(), DuckDuckGoSearchTool()],
            requirements=[
                ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False),
                ConditionalRequirement(
                    DuckDuckGoSearchTool, max_invocations=2, consecutive_allowed=False,
                    custom_checks=[lambda state: not is_casual(user_msg)]
                )
            ],
            instructions="""You are a helpful AI assistant. For search results, ALWAYS use proper markdown citations: [description](URL).

Examples:
- [OpenAI releases GPT-5](https://example.com/gpt5)
- [AI adoption increases 67%](https://example.com/ai-study)

Use DuckDuckGo for current info, facts, and specific questions. Respond naturally to casual greetings without search."""
        )
        
        yield trajectory.trajectory_metadata(
            title="Agent Ready",
            content="🛠️ Granite Chat ready with Think and Search tools"
        )
        
        response_text = ""
        search_results = None
        
        async for event, meta in agent.run(
            user_msg,
            execution=AgentExecutionConfig(max_iterations=20, max_retries_per_step=2, total_max_retries=5),
            expected_output="Markdown format with proper [text](URL) citations for search results."
        ):
            if meta.name == "success" and event.state.steps:
                step = event.state.steps[-1]
                if not step.tool:
                    continue
                    
                tool_name = step.tool.name
                
                if tool_name == "final_answer":
                    response_text += step.input["response"]
                elif tool_name == "DuckDuckGo":
                    search_results = getattr(step.output, 'results', None)
                    query = step.input.get("query", "Unknown")
                    count = len(search_results) if search_results else 0
                    
                    yield trajectory.trajectory_metadata(
                        title="DuckDuckGo Search",
                        content=f"🔍 Searched: '{query}' → {count} results"
                    )
                elif tool_name == "think":
                    yield trajectory.trajectory_metadata(
                        title="Thought",
                        content=step.input["thoughts"]
                    )
        
        await memory.add(AssistantMessage(response_text))
        
        citations, clean_text = extract_citations(response_text, search_results)

        yield clean_text
        
        if citations:
            citation_objects = []
            for cit in citations:
                citation_objects.append({
                    "url": cit["url"],
                    "title": cit["title"],
                    "description": cit["description"],
                    "start_index": cit["start_index"],
                    "end_index": cit["end_index"]
                })
            
            yield citation.citation_metadata(citations=citation_objects)
            
        yield trajectory.trajectory_metadata(
            title="Completion",
            content="✅ Response completed"
        )
        
    except Exception as e:
        print(f"❌ Error: {e}\n{traceback.format_exc()}")
        yield trajectory.trajectory_metadata(
            title="Error",
            content=f"❌ Error: {e}"
        )
        yield f"🚨 Error processing request: {e}"

def run():
    """Start the server"""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()