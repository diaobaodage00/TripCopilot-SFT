import json
import random
from pathlib import Path
cities = ["新加坡","东京","大阪","首尔","香港","上海","北京","杭州","深圳","曼谷"]
preferences_pool = [
    ["美食","城市观光"],
    ["购物","拍照"],
    ["历史文化","博物馆"],
    ["夜景","城市漫步"],
    ["自然风光","轻松休闲"],
    ["本地体验","街头美食"]
]

constraints_pool = [
    ["不要太赶"],
    ["预算尽量低"],
    ["公共交通优先"],
    ["少步行"],
    ["每天留休息时间"],
    ["可能下雨"],
    ["希望有雨天备选方案"],
    ["避免折返"],
    ["严格按固定结构输出"]
]

group_types = ["solo","couple","family","elderly","student","business"]
budget_levels = {
    "low":[1500,2000],
    "medium":[3000,4000,5000,6000],
    "high":[8000,10000]
}

def choose_budget():
    level = random.choice(list(budget_levels.keys()))
    amount = random.choice(budget_levels[level])
    return level,amount

def generate_instruction(city,days,budget,prefs,constraints,task):
    prefs_str="、".join(prefs)
    cons_str="，".join(constraints)

    if task=="plan_generation":
        return f"帮我规划一个{city}{days}日旅行，预算{budget}元，喜欢{prefs_str}，{cons_str}。"

    if task=="constraint_planning":
        return f"请帮我设计一个{city}{days}日旅行计划，预算{budget}元，偏好{prefs_str}，要求{cons_str}。"

    if task=="structured_generation":
        return f"请规划{city}{days}日旅行，预算{budget}元，偏好{prefs_str}。请严格按照“行程概览-Day1-Day2-预算建议-注意事项”的结构输出。"

    if task=="plan_revision":
        return f"这是我原来的{city}{days}日行程：Day1 安排很多景点；Day2 跨城市旅行；Day3 再安排远郊景点。请帮我优化这个计划并说明原因。"

def generate_output(city,days,budget,prefs):

    output=f"### 行程概览\n本次{days}日行程以{'、'.join(prefs)}为主题，整体节奏轻松。\n\n"

    for d in range(1,days+1):
        output+=f"### Day {d}\n上午：游览{city}核心景点。\n下午：体验当地特色街区。\n晚上：品尝当地美食。\n\n"

    output+=f"### 预算建议\n住宿约{int(budget*0.5)}元，餐饮约{int(budget*0.3)}元，交通及门票约{int(budget*0.2)}元。\n\n"

    output+="### 注意事项\n建议提前规划交通路线并关注天气变化。"

    return output

def generate_sample(i,task):

    city=random.choice(cities)
    days=random.choice([2,3,4])
    prefs=random.choice(preferences_pool)
    constraints=random.sample(constraints_pool,1)[0]

    budget_level,budget_amount=choose_budget()
    group=random.choice(group_types)

    instruction=generate_instruction(city,days,budget_amount,prefs,constraints,task)
    output=generate_output(city,days,budget_amount,prefs)

    sample={
        "id":f"travel_{i:04d}",
        "task_type":task,
        "city":city,
        "days":days,
        "budget_amount":budget_amount,
        "budget_level":budget_level,
        "group_type":group,
        "preferences":prefs,
        "constraints":constraints,
        "output_format":"markdown_itinerary",
        "language":"zh",
        "instruction":instruction,
        "output":output,
        "split":"train"
    }

    return sample


dataset=[]

tasks = (
    ["plan_generation"]*40 +
    ["constraint_planning"]*30 +
    ["plan_revision"]*15 +
    ["structured_generation"]*15
)

for i,task in enumerate(tasks,1):
    dataset.append(generate_sample(i,task))

output_path = Path(__file__).resolve().parent.parent / "processed" / "train_100.jsonl"
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf8") as f:
    for row in dataset:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Dataset generated: {output_path} ({len(dataset)} samples)")