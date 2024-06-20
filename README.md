<img width="100%" alt="SEACrowd Logo" src="https://github.com/SEACrowd/.github/blob/main/profile/assets/seacrowd-email-banner-without-logo.png?raw=true">

# Welcome to SEACrowd!

Southeast Asia is home to more than 1,000 native languages. Nevertheless, Southeast Asian NLP, vision-language, and speech processing is underrepresented in the research community, and one of the reasons is the lack of access to public datasets ([Aji et al., 2022](https://aclanthology.org/2022.acl-long.500/)). To address this issue, we initiate **SEACrowd**, a joint collaboration to collect NLP datasets for Southeast Asian languages. Help us collect and centralize Southeast Asian datasets, and be a co-author of our upcoming paper.

## How to Use

### Library Installation

Find seacrowd library (v0.1.3) at https://pypi.org/project/seacrowd/. (See our release notes [here](https://github.com/SEACrowd/seacrowd-datahub/releases/tag/0.1.3).)

To install SEACrowd, install the `seacrowd` package in your python environment via `pip`.

```
pip install seacrowd
```

### Using `seacrowd` library

To use the `seacrowd` package, simply import it in your code:
```
import seacrowd as sc
````

### List & Load Dataset
SEACrowd provides functions for listing and loading all datasets that are implemented in NusaCrowd
```
# List all datasets
dset_names = sc.list_datasets()

# List all datasets with their config names
dset_configs_dict = sc.list_datasets(with_config=True)

# Load a single dataset based on the dataset name
khpos_dset = sc.load_dataset("khpos", schema="seacrowd")

# Load multiple datasets based on the dataset names
dsets = sc.load_datasets(["thai_sum", "vsolscsum"], schema="seacrowd_t2t")
```

### List & Load Benchmark
In addition to dataset-related functions, SEACrowd provides additional functions for listing and loading some SEA benchmarks.
```
# List all benchmarks
benchmark_names = sc.list_benchmarks()

# Load all datasets in a benchmark
seacrowd_vl_dsets = sc.load_benchmark("SEACrowd-VL")
```

### Load Metadata
Aside from loading datasets and benchmarks, `seacrowd` also supports loading the metadata (e.g., license, description, citation,  etc.) of the dataloaders.
```
# Load metadata of a dataloader
khpos_meta = sc.for_dataset("khpos")

# Load metadata of multiple dataloaders
meta_dsets = sc.for_datasets(["thai_sum", "vsolscsum"])

# Load metadata of a config name
nusaparagraph_meta = sc.for_config_name("nusaparagraph_emot_jav_seacrowd_text")

# Load metadata of multiple config names
meta_dsets = sc.for_config_names(["sentiment_nathasa_review_seacrowd_text", "indonli_seacrowd_pairs"])
```

We can also load the dataloader from the metadata if we want.

```
# Load dataset from metadata
khpos_dset = khpos_meta.load_dataset()
```

> For the functions' sample outputs, check our [release notes](https://github.com/SEACrowd/seacrowd-datahub/releases/tag/0.1.3).

## How to Contribute

Check out our [CONTRIBUTING.md](https://github.com/SEACrowd/seacrowd-datahub/blob/master/CONTRIBUTING.md) for a gentle introduction to contributing in SEACrowd. Jump straight ahead to [DATALOADER.md](https://github.com/SEACrowd/seacrowd-datahub/blob/master/DATALOADER.md) if you have decided to contribute by implementing dataloaders for our Data Hub!

## Citation

If you are using any resources from SEACrowd, including datasheets, dataloaders, code, etc., please cite [the following publication](https://arxiv.org/pdf/2406.10118):

```
@article{lovenia2024seacrowd,
      title={SEACrowd: A Multilingual Multimodal Data Hub and Benchmark Suite for Southeast Asian Languages}, 
      author={Holy Lovenia and Rahmad Mahendra and Salsabil Maulana Akbar and Lester James V. Miranda and Jennifer Santoso and Elyanah Aco and Akhdan Fadhilah and Jonibek Mansurov and Joseph Marvin Imperial and Onno P. Kampman and Joel Ruben Antony Moniz and Muhammad Ravi Shulthan Habibi and Frederikus Hudi and Railey Montalan and Ryan Ignatius and Joanito Agili Lopo and William Nixon and Börje F. Karlsson and James Jaya and Ryandito Diandaru and Yuze Gao and Patrick Amadeus and Bin Wang and Jan Christian Blaise Cruz and Chenxi Whitehouse and Ivan Halim Parmonangan and Maria Khelli and Wenyu Zhang and Lucky Susanto and Reynard Adha Ryanda and Sonny Lazuardi Hermawan and Dan John Velasco and Muhammad Dehan Al Kautsar and Willy Fitra Hendria and Yasmin Moslem and Noah Flynn and Muhammad Farid Adilazuarda and Haochen Li and Johanes Lee and R. Damanhuri and Shuo Sun and Muhammad Reza Qorib and Amirbek Djanibekov and Wei Qi Leong and Quyet V. Do and Niklas Muennighoff and Tanrada Pansuwan and Ilham Firdausi Putra and Yan Xu and Ngee Chia Tai and Ayu Purwarianti and Sebastian Ruder and William Tjhi and Peerat Limkonchotiwat and Alham Fikri Aji and Sedrick Keh and Genta Indra Winata and Ruochen Zhang and Fajri Koto and Zheng-Xin Yong and Samuel Cahyawijaya},
      year={2024},
      eprint={2406.10118},
      journal={arXiv preprint arXiv: 2406.10118}
}
```

## Acknowledgements

Our initiative is heavily inspired by [NusaCrowd](https://github.com/IndoNLP/nusa-crowd/tree/master/nusacrowd) which provides open access data to 100+ Indonesian NLP corpora. You can check NusaCrowd paper (published in ACL Findings 2023) on the following [link](https://aclanthology.org/2023.findings-acl.868/).
