# Jenna's Granite Chat 🤖💬🧪

This is a research prototype of a general-purpose chat assistant built with the [BeeAI Framework](https://framework.beeai.dev/). It uses the experimental `RequirementAgent` pattern and is powered by a local or remote LLM configured via environment variables.

The assistant is designed for interactive use with tool-based reasoning and retrieval, session memory, and support for platform extensions like citation and trajectory tracking.

## Capabilities

* **Streaming chat** with memory
* **Tool invocation** based on conditional requirements:
  * `ThinkTool` (required at step 1 and after any tool)
  * `DuckDuckGoSearchTool` (max 2 invocations, skipped for casual messages)
* **Session-based memory** using `UnconstrainedMemory`
* **Citation extraction** via regex from markdown-style links
* **Structured trajectory metadata** for UI feedback
* **Basic error handling** with visible logs and UI feedback

## Running the Agent

To start the server:

```
uv run server
```

The server will start on the configured host and port.

## Functions

* `general_chat_assistant(...)`: Main entrypoint for the agent
* `RequirementAgent(...)`: Handles tool orchestration and memory
* `extract_citations(...)`: Extracts markdown-style citations into structured objects
* `is_casual(...)`: Lightweight filter to bypass tool use for simple greetings
* `run()`: Starts the agent server

## UI Extensions

* **CitationExtensionServer**: Enables structured citations for rendered links
* **TrajectoryExtensionServer**: Tracks each step of reasoning and tool usage for UI replay/debugging

## Sample Input/Output

**Input**:

> What are the latest advancements in AI research from 2025?

**Output**:

* Calls `ThinkTool`
* Calls `DuckDuckGoSearchTool` with relevant query
* Returns final response with `[label](url)` markdown
* Citations extracted and rendered in UI
* All steps logged via trajectory extension
