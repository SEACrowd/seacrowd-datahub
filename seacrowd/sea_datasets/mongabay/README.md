### PROVIDED DATA
- "mongabay-tag-classification"
- "mongabay-sentiment-classification"

### DATA CALLING EXAMPLE

- seacrowd format

    - mongabay-tag-classification
        ```
        from datasets import load_dataset

        data = load_dataset("seacrowd/sea_datasets/mongabay/mongabay.py", name="mongabay-tag-classification_seacrowd_t2t")

        >>> data["train"][0]
        {'id': '0', 'text_1': 'Pandemi, Momentum bagi Negara Serius Lindungi Hak Masyarakat Adat | ...', 'text_2': '[0.1111111119389534, 0.0, 0.0, 0.0, 0.1111111119389534, 0.0, 0.0, 0.0, 0.0, 0.1111111119389534, 0.1111111119389534, 0.0, 0.1111111119389534, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1111111119389534, 0.0, 0.1111111119389534, 0.0, 0.1111111119389534, 0.0, 0.0, 0.1111111119389534, 0.0, 0.0, 0.0]', 'text_1_name': 'text', 'text_2_name': 'weak_label'}
        ```

    - mongabay-sentiment-classification
        ```
        from datasets import load_dataset

        data = load_dataset("seacrowd/sea_datasets/mongabay/mongabay.py", name="mongabay-sentiment-classification_seacrowd_t2t")

        >>> data["train"][0]
        {'id': '0', 'text_1': 'Pandemi, Momentum bagi Negara Serius Lindungi Hak Masyarakat Adat | ...', 'text_2': '[1.0, 1.4414156535025313e-09, 1.320]', 'text_1_name': 'text', 'text_2_name': 'weak_label'}
        ```

- source format
  - mongabay-tag-classification
    ```
    from datasets import load_dataset

    data = load_dataset("seacrowd/sea_datasets/mongabay/mongabay.py", name="mongabay-tag-classification_source")

    data['train'][0]
    {'text': 'Pandemi, Momentum bagi Negara Serius Lindungi Hak Masyarakat Adat | ...', 'label': '[0.1111111119389534, 0.0, 0.0, 0.0, 0.1111111119389534, 0.0, 0.0, 0.0, 0.0, 0.1111111119389534, 0.1111111119389534, 0.0, 0.1111111119389534, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1111111119389534, 0.0, 0.1111111119389534, 0.0, 0.1111111119389534, 0.0, 0.0, 0.1111111119389534, 0.0, 0.0, 0.0]'}
    ```
  - mongabay-seniment-classification
    ```
    from datasets import load_dataset

    data = load_dataset("seacrowd/sea_datasets/mongabay/mongabay.py", name="mongabay-sentiment-classification_source")
    {'text': 'Pandemi, Momentum bagi Negara Serius Lindungi Hak Masyarakat Adat | ...', 'tags': "['Aparatur Sipil Negara' 'masyarakat desa' 'konflik' 'perusahaan' 'tambang']", 'label': '[1.0, 1.4414156535025313e-09, 1.3204033422198336e-09]'}
    ```
