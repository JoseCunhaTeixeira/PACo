"""The model PACo talks to, the same in both modes, workflow (paco.agent) and autonomous
(paco.autonomous), and as the evaluations' judge: its settings (PACO_LLM_*), and its clients,
PACo's ChatModel over OpenAI's chat API (a vLLM server, OpenAI, Gemini, any compatible server or
gateway) and over Claude's (Anthropic's API)."""
