import os
import re
import json
import csv
from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =========================
# 配置区
# =========================
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "outputs/qwen_travel_lora"
TEST_FILE = "data/processed/test.jsonl"
OUTPUT_DIR = "evaluation/results"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# 如果显存紧张，可以改成 True，强制半精度加载（前提是你的环境支持）
USE_TORCH_DTYPE_AUTO = True


# =========================
# 工具函数
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clean_generated_text(text: str) -> str:
    """
    去掉 prompt 前缀，只保留回答正文。
    """
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1].strip()

    # 去掉可能重复生成的 instruction 残留
    if "### Instruction:" in text and text.index("### Instruction:") < 20:
        parts = text.split("### Response:")
        if len(parts) > 1:
            text = parts[-1].strip()

    return text.strip()


def build_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def parse_constraints(item: Dict[str, Any]) -> List[str]:
    constraints = item.get("constraints", [])
    if isinstance(constraints, list):
        return [str(x).strip() for x in constraints if str(x).strip()]
    if isinstance(constraints, str):
        return [constraints.strip()] if constraints.strip() else []
    return []


def parse_preferences(item: Dict[str, Any]) -> List[str]:
    preferences = item.get("preferences", [])
    if isinstance(preferences, list):
        return [str(x).strip() for x in preferences if str(x).strip()]
    if isinstance(preferences, str):
        return [preferences.strip()] if preferences.strip() else []
    return []


# =========================
# 模型加载与生成
# =========================
class ModelRunner:
    def __init__(self, base_model_name: str, lora_path: str):
        self.base_model_name = base_model_name
        self.lora_path = lora_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if USE_TORCH_DTYPE_AUTO:
            model_kwargs["torch_dtype"] = "auto"

        print("[INFO] Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )
        self.base_model.eval()

        print("[INFO] Loading fine-tuned model (LoRA)...")
        self.finetuned_model = PeftModel.from_pretrained(
            self.base_model,
            self.lora_path
        )
        self.finetuned_model.eval()

    def generate_with_model(self, model, instruction: str) -> str:
        prompt = build_prompt(instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_generated_text(text)

    def generate_base(self, instruction: str) -> str:
        return self.generate_with_model(self.base_model, instruction)

    def generate_finetuned(self, instruction: str) -> str:
        return self.generate_with_model(self.finetuned_model, instruction)


# =========================
# 自动评估逻辑（规则版）
# =========================
def contains_section(text: str, title: str) -> bool:
    return title in text


def score_format(text: str, expected_days: int) -> Dict[str, Any]:
    """
    检查是否包含关键 section。
    """
    required_sections = [
        "### 行程概览",
        "### 预算建议",
        "### 注意事项",
    ]
    day_sections = [f"### Day {i}" for i in range(1, expected_days + 1)]

    matched = 0
    total = len(required_sections) + len(day_sections)

    section_hits = {}

    for sec in required_sections + day_sections:
        hit = contains_section(text, sec)
        section_hits[sec] = hit
        matched += int(hit)

    score = matched / total if total > 0 else 0.0
    return {
        "format_score": round(score, 4),
        "format_hit_count": matched,
        "format_total_count": total,
        "format_section_hits": section_hits,
    }


def score_days_reasonableness(text: str, expected_days: int) -> float:
    """
    如果要求 3 天，就应该至少有 Day 1~Day 3。
    """
    hits = 0
    for i in range(1, expected_days + 1):
        if f"### Day {i}" in text:
            hits += 1
    return round(hits / expected_days, 4) if expected_days > 0 else 0.0


def score_budget_mentions(text: str, budget_amount: int) -> Dict[str, Any]:
    """
    粗略看是否提到了预算 section 或金额相关词。
    """
    has_budget_section = "### 预算建议" in text
    has_money_word = any(word in text for word in ["元", "预算", "费用", "花费"])
    score = 1.0 if (has_budget_section and has_money_word) else 0.5 if (has_budget_section or has_money_word) else 0.0
    return {
        "budget_score": score,
        "has_budget_section": has_budget_section,
        "has_money_word": has_money_word,
    }


def score_preference_alignment(text: str, preferences: List[str]) -> Dict[str, Any]:
    if not preferences:
        return {
            "preference_score": 1.0,
            "preference_hit_count": 0,
            "preference_total_count": 0,
            "preference_hits": {},
        }

    hits = {}
    matched = 0
    for pref in preferences:
        hit = pref in text
        hits[pref] = hit
        matched += int(hit)

    score = matched / len(preferences)
    return {
        "preference_score": round(score, 4),
        "preference_hit_count": matched,
        "preference_total_count": len(preferences),
        "preference_hits": hits,
    }


def score_constraint_alignment(text: str, constraints: List[str]) -> Dict[str, Any]:
    """
    这是一个简化版规则评分。
    不是严格语义判断，但足够做项目第一版对比。
    """
    if not constraints:
        return {
            "constraint_score": 1.0,
            "constraint_hit_count": 0,
            "constraint_total_count": 0,
            "constraint_hits": {},
        }

    normalized_text = text.replace(" ", "").replace("\n", "")

    keyword_map = {
        "不要太赶": ["轻松", "宽松", "不赶", "休息"],
        "预算尽量低": ["经济", "低预算", "控制", "节省"],
        "公共交通优先": ["地铁", "公交", "公共交通"],
        "少步行": ["少步行", "打车", "接驳", "休息"],
        "每天留休息时间": ["休息", "午休", "休整"],
        "可能下雨": ["雨天", "室内", "备选"],
        "希望有雨天备选方案": ["雨天备选", "室内", "备选"],
        "避免折返": ["集中", "减少折返", "路线合理"],
        "严格按固定结构输出": ["### 行程概览", "### 预算建议", "### 注意事项"],
        "学生党": ["学生", "平价", "性价比"],
        "老人同行": ["老人", "少步行", "休息"],
        "带小孩": ["亲子", "小朋友", "休息"],
        "出差顺便玩": ["轻松", "半天", "短时间"],
        "晚上有半天空闲": ["晚上", "夜景", "晚餐"],
        "优化路线": ["优化", "合理", "减少折返"],
    }

    hits = {}
    matched = 0

    for cons in constraints:
        cons = cons.strip()
        keywords = keyword_map.get(cons, [cons])
        hit = any(kw.replace(" ", "") in normalized_text for kw in keywords)
        hits[cons] = hit
        matched += int(hit)

    score = matched / len(constraints)
    return {
        "constraint_score": round(score, 4),
        "constraint_hit_count": matched,
        "constraint_total_count": len(constraints),
        "constraint_hits": hits,
    }


def score_revision_task(text: str, task_type: str) -> Dict[str, Any]:
    """
    如果是 plan_revision，检查是否有“修改原因”或类似解释。
    """
    if task_type != "plan_revision":
        return {
            "revision_score": 1.0,
            "has_revision_reason": True,
        }

    has_reason = ("### 修改原因" in text) or ("修改原因" in text) or ("优化原因" in text) or ("原因" in text)
    return {
        "revision_score": 1.0 if has_reason else 0.0,
        "has_revision_reason": has_reason,
    }


def detect_obvious_hallucination(text: str, city: str) -> Dict[str, Any]:
    """
    简单检查是否出现一些常见不合理跨城/跨国词。
    这不是严格事实校验，只做项目第一版的轻量规则。
    """
    suspicious_map = {
        "新加坡": ["马六甲", "吉隆坡", "东京", "大阪"],
        "东京": ["新加坡", "曼谷", "香港"],
        "大阪": ["新加坡", "首尔", "曼谷"],
        "香港": ["东京", "首尔", "巴黎"],
        "上海": ["大阪", "曼谷", "首尔"],
        "北京": ["大阪", "新加坡", "曼谷"],
    }

    suspicious_terms = suspicious_map.get(city, [])
    found = [w for w in suspicious_terms if w in text]

    return {
        "hallucination_flag": len(found) > 0,
        "hallucination_terms": found,
        "hallucination_score": 0.0 if found else 1.0,
    }


def evaluate_single_output(item: Dict[str, Any], text: str) -> Dict[str, Any]:
    expected_days = int(item.get("days", 1))
    budget_amount = int(item.get("budget_amount", 0))
    preferences = parse_preferences(item)
    constraints = parse_constraints(item)
    task_type = item.get("task_type", "")
    city = item.get("city", "")

    fmt = score_format(text, expected_days)
    day_score = score_days_reasonableness(text, expected_days)
    budget = score_budget_mentions(text, budget_amount)
    pref = score_preference_alignment(text, preferences)
    cons = score_constraint_alignment(text, constraints)
    revision = score_revision_task(text, task_type)
    hallucination = detect_obvious_hallucination(text, city)

    overall = (
        0.30 * fmt["format_score"] +
        0.15 * day_score +
        0.15 * budget["budget_score"] +
        0.15 * pref["preference_score"] +
        0.15 * cons["constraint_score"] +
        0.05 * revision["revision_score"] +
        0.05 * hallucination["hallucination_score"]
    )

    return {
        "overall_score": round(overall, 4),
        "format_score": fmt["format_score"],
        "day_score": day_score,
        "budget_score": budget["budget_score"],
        "preference_score": pref["preference_score"],
        "constraint_score": cons["constraint_score"],
        "revision_score": revision["revision_score"],
        "hallucination_score": hallucination["hallucination_score"],
        "details": {
            **fmt,
            **budget,
            **pref,
            **cons,
            **revision,
            **hallucination,
        }
    }


def average_score(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(r[key] for r in rows) / len(rows), 4)


# =========================
# 主流程
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    print(f"[INFO] Loading test data from: {TEST_FILE}")
    test_data = load_jsonl(TEST_FILE)
    print(f"[INFO] Loaded {len(test_data)} test samples")

    runner = ModelRunner(BASE_MODEL_NAME, LORA_PATH)

    comparison_rows = []
    base_eval_rows = []
    finetuned_eval_rows = []

    for idx, item in enumerate(test_data, start=1):
        sample_id = item.get("id", f"sample_{idx}")
        instruction = item["instruction"]

        print(f"\n[INFO] Processing {idx}/{len(test_data)} - {sample_id}")

        try:
            base_output = runner.generate_base(instruction)
        except Exception as e:
            print(f"[ERROR] Base model generation failed for {sample_id}: {e}")
            base_output = f"[GENERATION_ERROR] {e}"

        try:
            finetuned_output = runner.generate_finetuned(instruction)
        except Exception as e:
            print(f"[ERROR] Fine-tuned model generation failed for {sample_id}: {e}")
            finetuned_output = f"[GENERATION_ERROR] {e}"

        base_eval = evaluate_single_output(item, base_output)
        finetuned_eval = evaluate_single_output(item, finetuned_output)

        base_eval_rows.append(base_eval)
        finetuned_eval_rows.append(finetuned_eval)

        winner = "tie"
        if finetuned_eval["overall_score"] > base_eval["overall_score"]:
            winner = "finetuned"
        elif finetuned_eval["overall_score"] < base_eval["overall_score"]:
            winner = "base"

        row = {
            "id": sample_id,
            "task_type": item.get("task_type", ""),
            "city": item.get("city", ""),
            "days": item.get("days", ""),
            "budget_amount": item.get("budget_amount", ""),
            "group_type": item.get("group_type", ""),
            "instruction": instruction,
            "reference_output": item.get("output", ""),
            "base_output": base_output,
            "finetuned_output": finetuned_output,
            "base_overall_score": base_eval["overall_score"],
            "finetuned_overall_score": finetuned_eval["overall_score"],
            "base_format_score": base_eval["format_score"],
            "finetuned_format_score": finetuned_eval["format_score"],
            "base_constraint_score": base_eval["constraint_score"],
            "finetuned_constraint_score": finetuned_eval["constraint_score"],
            "base_preference_score": base_eval["preference_score"],
            "finetuned_preference_score": finetuned_eval["preference_score"],
            "base_hallucination_score": base_eval["hallucination_score"],
            "finetuned_hallucination_score": finetuned_eval["hallucination_score"],
            "winner": winner,
            "base_details": base_eval["details"],
            "finetuned_details": finetuned_eval["details"],
        }
        comparison_rows.append(row)

    summary = {
        "num_samples": len(comparison_rows),
        "base_avg_overall_score": average_score(base_eval_rows, "overall_score"),
        "finetuned_avg_overall_score": average_score(finetuned_eval_rows, "overall_score"),
        "base_avg_format_score": average_score(base_eval_rows, "format_score"),
        "finetuned_avg_format_score": average_score(finetuned_eval_rows, "format_score"),
        "base_avg_constraint_score": average_score(base_eval_rows, "constraint_score"),
        "finetuned_avg_constraint_score": average_score(finetuned_eval_rows, "constraint_score"),
        "base_avg_preference_score": average_score(base_eval_rows, "preference_score"),
        "finetuned_avg_preference_score": average_score(finetuned_eval_rows, "preference_score"),
        "base_avg_hallucination_score": average_score(base_eval_rows, "hallucination_score"),
        "finetuned_avg_hallucination_score": average_score(finetuned_eval_rows, "hallucination_score"),
        "base_win_count": sum(1 for r in comparison_rows if r["winner"] == "base"),
        "finetuned_win_count": sum(1 for r in comparison_rows if r["winner"] == "finetuned"),
        "tie_count": sum(1 for r in comparison_rows if r["winner"] == "tie"),
    }

    json_path = os.path.join(OUTPUT_DIR, "comparison.json")
    csv_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")

    save_json(json_path, comparison_rows)
    save_json(summary_path, summary)

    csv_rows = []
    for r in comparison_rows:
        csv_rows.append({
            "id": r["id"],
            "task_type": r["task_type"],
            "city": r["city"],
            "days": r["days"],
            "budget_amount": r["budget_amount"],
            "group_type": r["group_type"],
            "instruction": r["instruction"],
            "base_overall_score": r["base_overall_score"],
            "finetuned_overall_score": r["finetuned_overall_score"],
            "base_format_score": r["base_format_score"],
            "finetuned_format_score": r["finetuned_format_score"],
            "base_constraint_score": r["base_constraint_score"],
            "finetuned_constraint_score": r["finetuned_constraint_score"],
            "base_preference_score": r["base_preference_score"],
            "finetuned_preference_score": r["finetuned_preference_score"],
            "base_hallucination_score": r["base_hallucination_score"],
            "finetuned_hallucination_score": r["finetuned_hallucination_score"],
            "winner": r["winner"],
            "base_output": r["base_output"],
            "finetuned_output": r["finetuned_output"],
        })

    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    save_csv(csv_path, csv_rows, fieldnames)

    print("\n[INFO] Done!")
    print(f"[INFO] comparison.json saved to: {json_path}")
    print(f"[INFO] comparison.csv saved to: {csv_path}")
    print(f"[INFO] summary.json saved to: {summary_path}")
    print("\n[INFO] Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()