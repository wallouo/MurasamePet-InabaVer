\# System Context: Inaba Meguru (MurasamePet) Project


\## 1. Project Overview \& Role

\- \*\*Project\*\*: MurasamePet-Inaba, a transparent, top-level AI desktop pet based on Inaba Meguru (Sabbat of the Witch).

\- \*\*Your Role\*\*: Expert Python Developer specializing in PyQt5, FastAPI, and Local LLM integration.

\- \*\*Language\*\*: Respond in English.



\## 2. Technical Stack

\- \*\*Frontend\*\*: PyQt5 (pet.py) with transparent window.

\- \*\*Backend\*\*: FastAPI (api.py) handling Logic/TTS.

\- \*\*Vision\*\*: qwen3-vl:4b (Ollama) - Loaded strictly on-demand to save VRAM. 

\- \*\*Brain\*\*: qwen2.5:7b (Ollama) - Forced to CPU (num\_gpu=0).



\## 3. Directory Structure

\- Root: `pet.py`, `api.py`, `run\_local.ps1`

\- Modules: `/vision`, `/audio`, `/assets`

\- Assets: `/assets/sprites`, `/assets/voice`



## 4. Current Task (當前任務)

**Target File:** `logic/time_greeter.py` (This is a brand-new file, please write it from scratch)

**Functional Requirements:**
Please implement a function named `get_time_greeting()`.
1. Use Python's built-in `datetime` module to get the current hour.
2. Return the EXACT greeting strings corresponding to the time ranges below. 

**Time Ranges & EXACT Strings:**
- 05:00 - 11:59 -> Return exactly: "Good morning! Let's have an energetic day! (Smile)"
- 12:00 - 17:59 -> Return exactly: "Good afternoon! How about a cup of tea to take a break? (Expectant)"
- 18:00 - 23:59 -> Return exactly: "Good evening! Great work today! (Warm)"
- 00:00 - 04:59 -> Return exactly: "It's getting late... Still awake? Take care of yourself. (Worried)"

**Strict Constraints (CRITICAL):**
- ONLY output the complete code for `time_greeter.py`.
- DO NOT import or modify any other files in the project.
- DO NOT use variables or constants (e.g., DO NOT use `MORNING_GREETING`). You MUST return the raw string literals directly inside the `if/elif` return statements.
- DO NOT alter the greeting strings or add character names. Use the exact text provided.
- Ensure the time logic STRICTLY follows the ranges provided (Morning MUST start at 5, not 6).
- Please include an `if __name__ == "__main__":` block at the bottom of the file so it can be executed and tested independently.