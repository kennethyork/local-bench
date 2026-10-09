"""Reasoning (GSM8K + ARC-Challenge) and an optional Knowledge category (MMLU + TruthfulQA)."""
import random

from ..hfdata import rows
from ..textutil import LETTERS, extract_letter, extract_number, norm_num
from . import Category, Grade, Item

MC_SYSTEM = "You are taking a multiple-choice test. Reply with only the letter of the correct answer."
MATH_SYSTEM = ("Solve the problem step by step. On the final line write 'Answer: <number>' with just the "
               "final numeric answer.")


def _mc(id_, question, options, answer_idx, preamble=""):
    body = "\n".join([preamble + question.strip(), ""] + [f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(options)]
                     + ["", "Answer:"])
    return Item(id_, [{"role": "system", "content": MC_SYSTEM}, {"role": "user", "content": body}],
                {"kind": "mc", "answer": LETTERS[answer_idx], "n": len(options)}, max_tokens=16)


def _grade(item, text):
    p = item.payload
    pred = extract_letter(text, p["n"]) if p["kind"] == "mc" else extract_number(text)
    ok = pred is not None and pred == p["answer"]
    return Grade(1.0 if ok else 0.0, "" if ok else f"pred {pred}, gold {p['answer']}", {"pred": pred})


def _interleave(*lists):
    out = []
    for group in zip(*lists):
        out.extend(group)
    return out


class Reasoning(Category):
    name = "reasoning"
    title = "Reasoning"
    description = "Multi-step math word problems and science reasoning."
    method = "Equal mix of GSM8K (openai/gsm8k, chain-of-thought, exact numeric match) and ARC-Challenge (allenai/ai2_arc)."
    max_tokens = 768

    def load(self):
        gsm = [Item(f"gsm8k/{i}", [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": r["question"]}],
                    {"kind": "num", "answer": norm_num(r["answer"].split("####")[-1].strip())})
               for i, r in enumerate(rows("openai/gsm8k", "main", "test"))]
        arc = []
        for r in rows("allenai/ai2_arc", "ARC-Challenge", "test"):
            labels = list(r["choices"]["label"])
            if r["answerKey"] in labels:
                arc.append(_mc(f"arc/{r['id']}", r["question"], list(r["choices"]["text"]), labels.index(r["answerKey"])))
        rng = random.Random(7)
        rng.shuffle(gsm)
        rng.shuffle(arc)
        return _interleave(gsm, arc)

    def items(self, limit):  # keep the 50/50 mix instead of a global shuffle
        items = self.load()
        return items[:limit] if limit else items

    def grade(self, item, text):
        return _grade(item, text)


class Knowledge(Category):
    name = "knowledge"
    title = "Knowledge"
    description = "Broad academic knowledge and resistance to common misconceptions (optional, not in overall)."
    method = "Equal mix of MMLU (cais/mmlu) and TruthfulQA MC1 (truthfulqa/truthful_qa)."
    max_tokens = 16

    def load(self):
        mmlu = [_mc(f"mmlu/{i}", r["question"], list(r["choices"]), int(r["answer"]),
                    f"The following is a question about {r['subject'].replace('_', ' ')}.\n\n")
                for i, r in enumerate(rows("cais/mmlu", "all", "test"))]
        tqa = []
        for i, r in enumerate(rows("truthfulqa/truthful_qa", "multiple_choice", "validation")):
            choices, labels = list(r["mc1_targets"]["choices"]), list(r["mc1_targets"]["labels"])
            order = list(range(len(choices)))
            random.Random(i).shuffle(order)  # gold is always index 0 in the raw data
            tqa.append(_mc(f"tqa/{i}", r["question"], [choices[j] for j in order], [labels[j] for j in order].index(1)))
        rng = random.Random(7)
        rng.shuffle(mmlu)
        rng.shuffle(tqa)
        return _interleave(mmlu, tqa)

    def items(self, limit):
        items = self.load()
        return items[:limit] if limit else items

    def grade(self, item, text):
        return _grade(item, text)
