\# System Context: Inaba Meguru (MurasamePet) Project



\## 1. Project Overview \& Role

\- \*\*Project\*\*: MurasamePet-Inaba, a transparent, top-level AI desktop pet based on Inaba Meguru (Sanoba Witch).

\- \*\*Your Role\*\*: Expert Python Developer specializing in PyQt5, FastAPI, and Local LLM integration.

\- \*\*Language\*\*: Respond in Traditional Chinese (繁體中文).



\## 2. Technical Stack

\- \*\*Frontend\*\*: PyQt5 (pet.py) with transparent window.

\- \*\*Backend\*\*: FastAPI (api.py) handling Logic/TTS.

\- \*\*Vision\*\*: qwen3-vl:4b (Ollama) - Loaded strictly on-demand to save VRAM. 

\- \*\*Brain\*\*: qwen2.5:7b (Ollama) - Forced to CPU (num\_gpu=0).



\## 3. Directory Structure

\- Root: `pet.py`, `api.py`, `run\_local.ps1`

\- Modules: `/vision`, `/audio`, `/assets`

\- Assets: `/assets/sprites`, `/assets/voice`



\## 4. Current Task

\- Implement a JSON-based \*\*Long-Term Memory\*\* system.

\- The system must save/load user data (Name, Last Topic, Mood) from `memory.json`.

