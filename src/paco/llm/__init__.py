"""The model an agent talks to, one setup for both of PACo's agents, the guided workflow
(paco.agent) and the autonomous geophysicist (paco.geo): its settings (PACO_LLM_*, the evaluations'
judge PACO_JUDGE_*), and its clients, PACo's ChatModel over OpenAI's chat API (OpenAI, Azure,
Gemini, Mistral, vLLM, any compatible server) and over Claude's (Anthropic's API, Vertex AI,
Bedrock, Foundry)."""
