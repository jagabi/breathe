import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "generate" / "sft"))
sys.path.insert(0, str(ROOT / "generate" / "sapo"))
from q_generator import generate as generate_q
from a_generator import generate as generate_a
from qa_generator import generate as generate_sapo


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_records(output_path: Path, domain: str) -> list:
    records = []
    if not output_path.exists():
        return records
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["domain"] == domain:
                records.append(item)
    return records


def format_few_shots(few_shots: list) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    parts = []
    for i, shot in enumerate(few_shots):
        parts.append(f"[Example {labels[i]}]\n{shot}")
    return "\n\n".join(parts)


def format_history(records: list) -> str:
    if not records:
        return ""
    parts = []
    for r in records:
        parts.append(f"---\nTask {r['index']}:\n{r['question']}\n---")
    return "\n".join(parts)


def parse_tasks(response_text: str) -> list:
    # --- is used as a separator between tasks, not a closing tag.
    # Capture from "Task N:" up to the next "---" boundary or end of string.
    pattern = r"Task\s+(\d+):\s*\n([\s\S]*?)(?=\n\s*---|\Z)"
    matches = re.findall(pattern, response_text)
    tasks = []
    for number, description in matches:
        tasks.append({
            "index": int(number),
            "question": description.strip(),
        })
    return tasks


def run_question_generation():
    yaml_path = ROOT / "generate" / "sft" / "q_gen.yaml"
    json_path = ROOT / "generate" / "sft" / "q_shots_per_domains.json"
    output_path = ROOT / "dataset" / "sft" / "train.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_yaml(yaml_path)
    system_prompt: str = config["system"].strip()
    user_template: str = config["user"]

    domains_data = load_json(json_path)

    for domain_info in domains_data:
        domain: str = domain_info["domain"]
        few_shot_text = format_few_shots(domain_info["few-shot"])

        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        for call_idx in range(5):
            existing = load_existing_records(output_path, domain)
            if existing and max(r["index"] for r in existing) >= 100:
                print(f"[SKIP] {domain} already has 100 tasks, moving to next domain")
                break

            history_text = format_history(existing)

            user_prompt = user_template
            user_prompt = user_prompt.replace("{{DOMAIN}}", domain)
            user_prompt = user_prompt.replace("{{few-shot}}", few_shot_text)
            user_prompt = user_prompt.replace("{{history}}", history_text)

            next_task_num = len(existing) + 1
            print(f"\n[Call {call_idx + 1}/5 | Domain: {domain} | Tasks so far: {len(existing)} | Next starts at Task {next_task_num}]")

            response = generate_q(system_prompt, user_prompt)

            # ===== DEBUG_PRINT START (remove after verifying format) =====
            print("\n" + "-" * 40 + " FULL RESPONSE " + "-" * 40)
            print(response)
            print("-" * 95)
            # ===== DEBUG_PRINT END =====

            tasks = parse_tasks(response)
            print(f"Parsed {len(tasks)} tasks from response")

            if not tasks:
                print("[WARNING] No tasks parsed - check response format above")
                continue

            with open(output_path, "a", encoding="utf-8") as f:
                for task in tasks:
                    record = {
                        "domain": domain,
                        "index": task["index"],
                        "system": "",
                        "question": task["question"],
                        "answer": "",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Saved {len(tasks)} tasks → {output_path}")


def run_answer_generation():
    yaml_path = ROOT / "generate" / "sft" / "a_gen.yaml"
    output_path = ROOT / "dataset" / "sft" / "train.jsonl"

    if not output_path.exists():
        print("[ERROR] train.jsonl not found. Run --qa q first.")
        return

    config = load_yaml(yaml_path)
    system_prompt: str = config["system"].strip()
    user_template: str = config["user"]

    records = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    pending = [r for r in records if not r.get("answer")]
    print(f"Total rows: {len(records)} | Pending answers: {len(pending)}")

    index_map = {(r["domain"], r["index"]): i for i, r in enumerate(records)}
    file_lock = threading.Lock()
    completed_count = 0

    def process_record(record):
        user_prompt = user_template.replace("{{task}}", record["question"])
        response = generate_a(system_prompt, user_prompt)
        return record, response

    def save_answer(record, response):
        nonlocal completed_count
        record["answer"] = response
        records[index_map[(record["domain"], record["index"])]] = record
        completed_count += 1

        # ===== DEBUG_PRINT START (remove after verifying format) =====
        print(f"\n[{completed_count}/{len(pending)}] domain={record['domain']} index={record['index']}")
        print("-" * 40 + " FULL RESPONSE " + "-" * 40)
        print(response)
        print("-" * 95)
        # ===== DEBUG_PRINT END =====

        with file_lock:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved answer for domain={record['domain']} index={record['index']}")

    BATCH_SIZE = 50
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        print(f"\n[Batch {batch_start // BATCH_SIZE + 1}] Sending {len(batch)} requests concurrently...")

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {executor.submit(process_record, r): r for r in batch}
            for future in as_completed(futures):
                record, response = future.result()
                save_answer(record, response)

    print(f"\nDone. Answers written to {output_path}")


def extract_json_objects(text: str) -> list:
    """Extract top-level JSON objects from a string by tracking brace depth."""
    objects = []
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


def parse_sapo_response(response_text: str) -> list:
    text = response_text.strip()
    text = re.sub(r'^```(?:json)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [{"question": item["prompt"], "checklist": item["checklist"]} for item in data]
    except (json.JSONDecodeError, KeyError):
        pass

    objects = extract_json_objects(text)
    results = []
    for obj in objects:
        if "prompt" in obj and "checklist" in obj:
            results.append({"question": obj["prompt"], "checklist": obj["checklist"]})
    return results


def load_sapo_records(output_path: Path) -> list:
    records = []
    if not output_path.exists():
        return records
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_sapo_sorted(output_path: Path, records: list):
    records.sort(key=lambda r: (r["category"], r["index"]))
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_sapo_generation(category_id: int, count: int):
    yaml_path = ROOT / "generate" / "sapo" / "qa.yaml"
    cat_path = ROOT / "generate" / "sapo" / "category.json"
    output_path = ROOT / "dataset" / "sapo" / "train.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_yaml(yaml_path)
    system_prompt: str = config["system"].strip()
    user_template: str = config["user"]

    with open(cat_path, "r", encoding="utf-8") as f:
        cat_map = json.load(f)

    category_key = str(category_id)
    if category_key not in cat_map:
        print(f"[ERROR] Unknown category id: {category_id}. Available: {list(cat_map.keys())}")
        return

    category_name = cat_map[category_key]
    print(f"Category: {category_name} (id={category_id})")
    print(f"Generating {count} QA pairs...")

    existing = load_sapo_records(output_path)
    existing_for_cat = [r for r in existing if r["category"] == category_name]

    history_text = ""
    if existing_for_cat:
        history_lines = ["Previously generated prompts for this category (DO NOT duplicate):"]
        for r in existing_for_cat:
            history_lines.append(f"- {r['question'][:200]}")
        history_text = "\n".join(history_lines)

    user_prompt = user_template
    user_prompt = user_prompt.replace("{{category}}", category_name)
    user_prompt = user_prompt.replace("{{count}}", str(count))
    user_prompt = user_prompt.replace("{{history}}", history_text)

    response = generate_sapo(system_prompt, user_prompt)

    # ===== DEBUG_PRINT START (remove after verifying format) =====
    print("\n" + "-" * 40 + " FULL RESPONSE " + "-" * 40)
    print(response)
    print("-" * 95)
    # ===== DEBUG_PRINT END =====

    pairs = parse_sapo_response(response)
    print(f"Parsed {len(pairs)} QA pairs from response")

    if not pairs:
        print("[WARNING] No QA pairs parsed - check response format above")
        return

    max_index = 0
    for r in existing_for_cat:
        max_index = max(max_index, r["index"])

    new_records = []
    for i, pair in enumerate(pairs):
        new_records.append({
            "category": category_name,
            "index": max_index + i + 1,
            "question": pair["question"],
            "checklist": pair["checklist"],
        })

    all_records = existing + new_records
    save_sapo_sorted(output_path, all_records)
    print(f"Saved {len(new_records)} QA pairs → {output_path} (total: {len(all_records)})")


def main():
    parser = argparse.ArgumentParser(description="Dataset generator for breathe project")
    parser.add_argument("--type", type=str, required=True, choices=["sft", "rl", "sapo"], help="Dataset type")
    parser.add_argument("--qa", type=str, choices=["q", "a"], help="Generate questions or answers (sft only)")
    parser.add_argument("--category", type=int, help="Category id (sapo only)")
    parser.add_argument("--count", type=int, help="Number of QA pairs to generate (sapo only)")
    args = parser.parse_args()

    if args.type == "sft":
        if not args.qa:
            parser.error("--qa is required for --type sft")
        if args.qa == "q":
            run_question_generation()
        elif args.qa == "a":
            run_answer_generation()
    elif args.type == "sapo":
        if args.category is None or args.count is None:
            parser.error("--category and --count are required for --type sapo")
        run_sapo_generation(args.category, args.count)
    else:
        print(f"Not implemented yet: --type {args.type}")


if __name__ == "__main__":
    main()
