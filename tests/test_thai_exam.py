"""Offline regression tests for the ThaiExam loader."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from seacrowd.sea_datasets.thai_exam.thai_exam import ThaiExamDataset
from seacrowd.utils import schemas


class TestThaiExam(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def builder(self, subset, schema="seacrowd_qa"):
        return ThaiExamDataset(config_name=f"thai_exam_{subset}_{schema}", cache_dir=str(self.root / "cache"))

    def generate(self, builder, row, split="test"):
        path = self.root / "sample.jsonl"
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        return list(builder._generate_examples(path, split))

    def row(self, subset):
        row = {"question": "คำถาม", "subject": "thai", "a": "หนึ่ง", "b": "สอง", "c": "สาม", "d": "สี่", "answer": "b"}
        if subset != "ic":
            row["e"] = "ห้า"
        if subset == "onet":
            row.update(no=3.1, year=2566)
        return row

    def test_source_preserves_fields_and_types(self):
        for subset in ("onet", "ic", "tgat", "tpat1", "a_level"):
            with self.subTest(subset=subset):
                row = self.row(subset)
                builder = self.builder(subset, "source")
                _, result = self.generate(builder, row)[0]
                self.assertEqual(builder.info.features.encode_example(result), row)
                self.assertEqual(set(builder.info.features), set(row))

    def test_choices_answers_and_metadata(self):
        for subset in ("onet", "ic", "tgat", "tpat1", "a_level"):
            with self.subTest(subset=subset):
                row = self.row(subset)
                row["answer"] = "d" if subset == "ic" else "e"
                builder = self.builder(subset)
                _, result = self.generate(builder, row)[0]
                self.assertEqual(result["choices"], [row[k] for k in "abcde" if k in row])
                self.assertEqual(result["answer"], [row[row["answer"]]])
                self.assertEqual(result["meta"], {k: row[k] for k in ("subject", "no", "year") if k in row})
                self.assertEqual(result["type"], "multiple_choice")
                self.assertEqual(result["context"], "")
                self.assertEqual(result["document_id"], "")

    def test_empty_fifth_choice_is_omitted(self):
        row = self.row("tgat")
        row["e"] = ""
        _, result = self.generate(self.builder("tgat"), row)[0]
        self.assertEqual(result["choices"], [row[k] for k in "abcd"])
        self.assertEqual(result["answer"], [row["b"]])

    def test_invalid_answers_fail_with_record_id(self):
        for answer in ("e", "z", "", "ab", None):
            with self.subTest(answer=answer):
                row = self.row("ic")
                row["answer"] = answer
                with self.assertRaisesRegex(ValueError, "thai_exam_ic_test_0"):
                    self.generate(self.builder("ic"), row)

    def test_ids_are_stable_and_distinct_across_splits_and_subsets(self):
        ids = set()
        for subset in ("onet", "ic", "tgat", "tpat1", "a_level"):
            builder = self.builder(subset)
            for split in ("train", "test"):
                first = self.generate(builder, self.row(subset), split)
                self.assertEqual(first, self.generate(builder, self.row(subset), split))
                key, example = first[0]
                self.assertEqual(key, example["id"])
                self.assertEqual(key, example["question_id"])
                self.assertNotIn(key, ids)
                ids.add(key)

    def test_schema_isolation(self):
        original = deepcopy(schemas.qa_features)
        onet = self.builder("onet")
        ic = self.builder("ic")
        onet.info.features["meta"].pop("subject")
        self.assertIn("subject", ic.info.features["meta"])
        self.assertEqual(schemas.qa_features, original)


if __name__ == "__main__":
    unittest.main()
