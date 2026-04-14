#!/usr/bin/env python3
"""Generate RULER Needle-in-a-Haystack (NIAH) test data at various context lengths.

Produces JSONL files under data/ruler_niah/ that the evals/ruler_niah*.yaml
task configs consume via lm-evaluation-harness.

Usage:
    python3 gen_ruler_niah.py                        # all default lengths
    python3 gen_ruler_niah.py --lengths 4096 16384   # specific lengths only
    python3 gen_ruler_niah.py --samples 200          # more samples per length
"""

import argparse
import json
import os
import random
import string
import textwrap

SEED = 42

DEFAULT_LENGTHS = [4096, 8192, 16384, 32768, 65536]
DEFAULT_SAMPLES = 100

CHARS_PER_TOKEN = 4  # conservative English approximation

NEEDLE_TEMPLATE = "The special key is {value}. Remember this."
QUERY = "What is the special key mentioned in the text above? Respond with only the key value, nothing else."

NOISE_SENTENCES = [
    "The weather today is partly cloudy with a chance of afternoon showers.",
    "Research indicates that regular exercise improves cognitive function significantly.",
    "The quarterly report shows steady growth across all major market segments.",
    "Modern architecture emphasizes sustainable materials and energy efficiency.",
    "The committee reviewed the proposal and requested additional documentation.",
    "Ocean currents play a critical role in regulating global climate patterns.",
    "The software update includes several performance improvements and bug fixes.",
    "Historical records suggest the region was inhabited as early as 3000 BCE.",
    "Advances in battery technology are accelerating the adoption of electric vehicles.",
    "The library recently expanded its digital collection to over two million titles.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The new policy framework aims to reduce administrative overhead by thirty percent.",
    "Archaeological excavations uncovered artifacts dating to the Bronze Age.",
    "Machine learning models require large datasets for effective training.",
    "The bridge renovation project is expected to be completed by next quarter.",
    "Coral reefs support approximately twenty-five percent of all marine species.",
    "The audit identified several areas where operational efficiency could improve.",
    "Gravitational waves were first directly detected in September 2015.",
    "The conference attracted over three thousand participants from forty countries.",
    "Renewable energy sources now account for a growing share of electricity generation.",
    "The study found a strong correlation between sleep quality and productivity.",
    "Urban planning increasingly prioritizes pedestrian-friendly infrastructure.",
    "The protein structure was determined using cryo-electron microscopy techniques.",
    "Supply chain disruptions led to increased costs across multiple industries.",
    "The algorithm processes input data in linear time relative to array size.",
    "Volcanic activity along the Pacific Ring of Fire remains closely monitored.",
    "The festival celebrates traditional music and dance from the region.",
    "Quantum computing promises exponential speedups for certain problem classes.",
    "The survey revealed that customer satisfaction improved after the redesign.",
    "Biodiversity loss is one of the most pressing environmental challenges today.",
    "The telescope captured images of a galaxy twelve billion light-years away.",
    "Effective communication is essential for successful project management.",
    "The river delta ecosystem supports hundreds of migratory bird species.",
    "Cloud computing has transformed how organizations manage IT infrastructure.",
    "The treaty established guidelines for international maritime navigation.",
    "Fermentation has been used in food preservation for thousands of years.",
    "The startup secured Series B funding to expand into European markets.",
    "Tectonic plate movement causes earthquakes along major fault lines.",
    "The curriculum was updated to include data literacy and computational thinking.",
    "Antibiotics revolutionized medicine but require careful stewardship.",
    "The park covers over five hundred square kilometers of protected wilderness.",
    "Neural networks loosely model the structure of biological brain circuits.",
    "The manufacturing process uses recycled materials wherever possible.",
    "Migratory patterns of monarch butterflies span thousands of kilometers.",
    "The team deployed a microservices architecture to improve scalability.",
    "Glacial retreat has accelerated significantly over the past two decades.",
    "The museum houses one of the largest collections of impressionist paintings.",
    "Distributed systems must handle network partitions and eventual consistency.",
    "Pollinators are essential for the reproduction of many flowering plants.",
    "The financial model projects a return on investment within three years.",
]


def generate_needle_value(rng: random.Random) -> str:
    """Random 5-8 digit numeric string."""
    length = rng.randint(5, 8)
    return "".join(rng.choices(string.digits, k=length))


def build_filler(target_chars: int, rng: random.Random) -> list[str]:
    """Return a list of noise sentences totalling ~target_chars characters."""
    sentences: list[str] = []
    total = 0
    while total < target_chars:
        s = rng.choice(NOISE_SENTENCES)
        sentences.append(s)
        total += len(s) + 1  # +1 for joining space
    return sentences


def build_example(
    context_tokens: int,
    needle_depth_pct: float,
    rng: random.Random,
) -> dict:
    value = generate_needle_value(rng)
    needle = NEEDLE_TEMPLATE.format(value=value)

    needle_chars = len(needle) + 2  # surrounding spaces
    filler_budget = context_tokens * CHARS_PER_TOKEN - needle_chars - len(QUERY) - 50
    filler_budget = max(filler_budget, 200)

    sentences = build_filler(filler_budget, rng)
    insert_idx = max(1, int(len(sentences) * needle_depth_pct))
    sentences.insert(insert_idx, needle)

    context = " ".join(sentences)

    return {
        "context": context,
        "query": QUERY,
        "answer": value,
        "needle_depth_percent": int(needle_depth_pct * 100),
        "context_length_tokens": context_tokens,
    }


def generate_dataset(context_tokens: int, num_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    examples = []
    for i in range(num_samples):
        depth = depths[i % len(depths)]
        examples.append(build_example(context_tokens, depth, rng))
    return examples


def length_label(tokens: int) -> str:
    if tokens >= 1024:
        return f"{tokens // 1024}k"
    return str(tokens)


def main():
    parser = argparse.ArgumentParser(description="Generate RULER NIAH test data")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help=f"Context lengths in tokens (default: {DEFAULT_LENGTHS})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Samples per context length (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/ruler_niah/ relative to this script)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output_dir or os.path.join(script_dir, "data", "ruler_niah")
    os.makedirs(out_dir, exist_ok=True)

    for length in args.lengths:
        label = length_label(length)
        filename = f"niah_{label}.jsonl"
        path = os.path.join(out_dir, filename)

        examples = generate_dataset(length, args.samples, seed=args.seed + length)
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"  {path}  ({len(examples)} samples, ~{label} tokens)")

    print(f"\nDone. Data written to {out_dir}/")
    print("Now run evals with:  ./run.sh ... --task ruler_niah_4k --gen-max-tokens 4096 --eval-only")


if __name__ == "__main__":
    main()
