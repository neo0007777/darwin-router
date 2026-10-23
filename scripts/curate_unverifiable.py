#!/usr/bin/env python3
"""
Curate Non-verifiable Exploratory Set (Task 1.4)
Generates 30 subjective, creative, and open-ended queries (15 simple, 15 complex).
Saved to data/eval_unverifiable_30.jsonl.
Strictly labeled for observational tracking (no formal verifier accuracy claims).
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "eval_unverifiable_30.jsonl"

UNVERIFIABLE_QUERIES = [
    # ── 15 Simple Queries (format, tone, casual greeting, quick rewrite) ──
    {
        "id": "unverifiable_simple_01",
        "text": "Draft a polite 2-sentence note thanking a colleague for covering my shift yesterday.",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "polite_note",
    },
    {
        "id": "unverifiable_simple_02",
        "text": "Rewrite this sentence in a cheerful, energetic tone: 'The report has been completed.'",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "tone_shift",
    },
    {
        "id": "unverifiable_simple_03",
        "text": "Suggest 3 catchy subject lines for an internal company newsletter about wellness week.",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "brainstorming",
    },
    {
        "id": "unverifiable_simple_04",
        "text": "Summarize this in one sentence: 'The team decided to migrate from Jenkins to GitHub Actions after evaluating build times, runner costs, and developer ergonomics over the last two sprints.'",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "summarization",
    },
    {
        "id": "unverifiable_simple_05",
        "text": "Make this customer support response warmer and more empathetic: 'Your refund will process in 5 business days.'",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "empathy_refinement",
    },
    {
        "id": "unverifiable_simple_06",
        "text": "Translate 'Could we please reschedule our 2 PM check-in to tomorrow?' into informal French.",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "casual_translation",
    },
    {
        "id": "unverifiable_simple_07",
        "text": "Turn this raw meeting note into three clean bullet points: 'Need to update README by Friday, Shiva will test Ollama provider, and we should tag Filip on Discord once verified.'",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "formatting",
    },
    {
        "id": "unverifiable_simple_08",
        "text": "Draft an automated out-of-office message mentioning I will have limited connectivity until Monday.",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "template_draft",
    },
    {
        "id": "unverifiable_simple_09",
        "text": "Provide 3 conversational icebreakers for a virtual engineering team standup.",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "icebreakers",
    },
    {
        "id": "unverifiable_simple_10",
        "text": "Rephrase this feedback to be more constructive: 'Your slides were boring and had too much text.'",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "constructive_feedback",
    },
    {
        "id": "unverifiable_simple_11",
        "text": "Give me a brief welcoming intro message for a new Discord channel dedicated to hackathon tips.",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "welcome_message",
    },
    {
        "id": "unverifiable_simple_12",
        "text": "Shorten this paragraph by 50% while retaining the main message: 'We are reaching out to let you know that our scheduled maintenance window will commence at 2:00 AM UTC. During this brief period, some users might experience intermittent latency when querying historical log archives.'",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "condensation",
    },
    {
        "id": "unverifiable_simple_13",
        "text": "Suggest two alternative names for a developer tool called 'Darwin Router'.",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "naming",
    },
    {
        "id": "unverifiable_simple_14",
        "text": "Write a friendly closing sentence for a software proposal email.",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "closing_phrase",
    },
    {
        "id": "unverifiable_simple_15",
        "text": "Format this list into a clean Markdown table with columns Name, Role, and Location: Alice (Engineer, NYC), Bob (Product, SF), Charlie (Designer, London).",
        "difficulty": "simple",
        "heuristic_target": "simple",
        "intent": "table_formatting",
    },

    # ── 15 Complex Queries (nuanced analysis, multi-stakeholder trade-offs, synthesis) ──
    {
        "id": "unverifiable_complex_01",
        "text": "Analyze the trade-offs between deploying self-hosted quantized models vs relying exclusively on hosted frontier APIs for an early-stage B2B SaaS startup. Address latency, data privacy, capital expenditure, and talent constraints.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "strategic_tradeoffs",
    },
    {
        "id": "unverifiable_complex_02",
        "text": "Draft a multi-phase RFC proposal for migrating a monolithic distributed database to an event-sourced microservices architecture, accounting for dual-write data consistency and rollback strategies.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "architectural_rfc",
    },
    {
        "id": "unverifiable_complex_03",
        "text": "Write a nuanced comparative critique of Kantian deontological ethics versus Rule Utilitarianism when applied to autonomous vehicle moral dilemmas.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "philosophical_analysis",
    },
    {
        "id": "unverifiable_complex_04",
        "text": "Synthesize the competing arguments surrounding algorithmic copyright and training on publicly accessible copyrighted data. Cover fair use doctrines, market dilution, and opt-out frameworks.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "legal_policy_synthesis",
    },
    {
        "id": "unverifiable_complex_05",
        "text": "Construct an executive briefing outlining a risk mitigation framework for generative AI prompt injection in financial services, addressing indirect injection, data exfiltration, and human-in-the-loop auditability.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "security_briefing",
    },
    {
        "id": "unverifiable_complex_06",
        "text": "Draft an empathetic yet legally rigorous executive communication addressing staff layoffs resulting from an organizational restructuring and market downturn.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "crisis_communication",
    },
    {
        "id": "unverifiable_complex_07",
        "text": "Write a dialectical essay exploring whether mathematical objects are discovered or invented, evaluating formalist and platonist viewpoints.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "epistemology_essay",
    },
    {
        "id": "unverifiable_complex_08",
        "text": "Design a complete curriculum for an intensive 8-week bootcamp on Distributed Systems and Consensus protocols for senior backend engineers, with weekly learning goals, labs, and evaluation criteria.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "curriculum_design",
    },
    {
        "id": "unverifiable_complex_09",
        "text": "Analyze the macroeconomic implications of demographic aging in East Asian economies over the next three decades, specifically examining fiscal solvency, labor market automation, and healthcare consumption.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "macroeconomic_forecast",
    },
    {
        "id": "unverifiable_complex_10",
        "text": "Formulate a balanced post-incident postmortem analysis for a cascading multi-region cloud outage, explaining the root cause, contributing cultural factors, and defensive architectural changes.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "blameless_postmortem",
    },
    {
        "id": "unverifiable_complex_11",
        "text": "Evaluate the strengths and pitfalls of using synthetic data vs human-in-the-loop annotations for alignment of reasoning models, focusing on model collapse, reward hacking, and distributional drift.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "ai_research_critique",
    },
    {
        "id": "unverifiable_complex_12",
        "text": "Write a comprehensive venture capital investment memo evaluating a fictitious seed-stage company offering 'Zero-Knowledge Machine Learning Verification for Healthcare'.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "vc_investment_memo",
    },
    {
        "id": "unverifiable_complex_13",
        "text": "Compare and contrast the Byzantine fault tolerance mechanisms of PBFT, Raft with witness nodes, and HotStuff in terms of communication complexity and view-change overhead.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "consensus_comparison",
    },
    {
        "id": "unverifiable_complex_14",
        "text": "Draft a detailed mediation proposal resolving a bitter IP and equity dispute between co-founders where the technical founder built the prototype before incorporation.",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "dispute_mediation",
    },
    {
        "id": "unverifiable_complex_15",
        "text": "Critique the architectural assumptions of Microkernel OS designs versus Monolithic Unix kernels in modern hyper-scaler environments with hardware acceleration (e.g., DPUs and GPUs).",
        "difficulty": "complex",
        "heuristic_target": "reasoning",
        "intent": "os_architecture_critique",
    },
]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in UNVERIFIABLE_QUERIES:
            entry = {
                "id": item["id"],
                "text": item["text"],
                "category": "unverifiable_exploratory",
                "difficulty": item["difficulty"],
                "label": item["heuristic_target"],
                "expected_route": item["heuristic_target"],
                "verifiable": False,
                "intent": item["intent"],
                "note": "Holdout query for subjective/exploratory observation. Excluded from quantitative accuracy claims.",
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {len(UNVERIFIABLE_QUERIES)} non-verifiable queries into {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
