import os
import regex as re

from huggingface_hub import HfApi
from io import BytesIO

_SEA_DATASETS_PATH = "./seacrowd/sea_datasets/"
_SEACROWD_CITATION = '''
@article{lovenia2024seacrowd,
    title={SEACrowd: A Multilingual Multimodal Data Hub and Benchmark Suite for Southeast Asian Languages}, 
    author={Holy Lovenia and Rahmad Mahendra and Salsabil Maulana Akbar and Lester James V. Miranda and Jennifer Santoso and Elyanah Aco and Akhdan Fadhilah and Jonibek Mansurov and Joseph Marvin Imperial and Onno P. Kampman and Joel Ruben Antony Moniz and Muhammad Ravi Shulthan Habibi and Frederikus Hudi and Railey Montalan and Ryan Ignatius and Joanito Agili Lopo and William Nixon and Börje F. Karlsson and James Jaya and Ryandito Diandaru and Yuze Gao and Patrick Amadeus and Bin Wang and Jan Christian Blaise Cruz and Chenxi Whitehouse and Ivan Halim Parmonangan and Maria Khelli and Wenyu Zhang and Lucky Susanto and Reynard Adha Ryanda and Sonny Lazuardi Hermawan and Dan John Velasco and Muhammad Dehan Al Kautsar and Willy Fitra Hendria and Yasmin Moslem and Noah Flynn and Muhammad Farid Adilazuarda and Haochen Li and Johanes Lee and R. Damanhuri and Shuo Sun and Muhammad Reza Qorib and Amirbek Djanibekov and Wei Qi Leong and Quyet V. Do and Niklas Muennighoff and Tanrada Pansuwan and Ilham Firdausi Putra and Yan Xu and Ngee Chia Tai and Ayu Purwarianti and Sebastian Ruder and William Tjhi and Peerat Limkonchotiwat and Alham Fikri Aji and Sedrick Keh and Genta Indra Winata and Ruochen Zhang and Fajri Koto and Zheng-Xin Yong and Samuel Cahyawijaya},
    year={2024},
    eprint={2406.10118},
    journal={arXiv preprint arXiv: 2406.10118}
}
'''

def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)

def construct_readme(dsetname):
    module_path = f"seacrowd.sea_datasets.{dsetname}.{dsetname}"

    dset_name = import_from(module_path, "_DATASETNAME")
    description = import_from(module_path, "_DESCRIPTION")
    homepage = import_from(module_path, "_HOMEPAGE")
    is_local = import_from(module_path, "_LOCAL")
    languages = import_from(module_path, "_LANGUAGES")
    supported_tasks = import_from(module_path, "_SUPPORTED_TASKS")
    source_version = import_from(module_path, "_SOURCE_VERSION")
    seacrowd_version = import_from(module_path, "_SEACROWD_VERSION")
    citation = import_from(module_path, "_CITATION")
    license = import_from(module_path, "_LICENSE")

    languages_part = "\n- " + "\n- ".join(languages)
    pretty_name_part = dset_name.replace("_", " ").title()
    task_categories_part = "\n- " + "\n- ".join(task.name.replace("_", "-").lower() for task in supported_tasks)
    if "(" in license and ")" in license:
        license_part = license[license.find("(")+1:license.find(")")]
        readme_string = f'\n---\nlicense: {license_part}\nlanguage: {languages_part}\npretty_name: {pretty_name_part}\ntask_categories: {task_categories_part}\ntags: {task_categories_part}\n---\n'
    else:
        readme_string = f'\n---\nlanguage: {languages_part}\npretty_name: {pretty_name_part}\ntask_categories: {task_categories_part}\ntags: {task_categories_part}\n---\n'
    readme_string += f'\n\n# {pretty_name_part}'
    readme_string += f'\n\n{description}'
    if is_local:
        readme_string += "\n\nThis is a local dataset. You have to obtain this dataset separately from [{homepage}]({homepage}) to use this dataloader."
    readme_string += f'\n\n## Languages\n\n{", ".join(languages)}'
    readme_string += f'\n\n## Supported Tasks\n\n{", ".join([str(task.name.replace("_", " ").title()) for task in supported_tasks])}'
    readme_string += f'''
    \n## Dataset Usage
    ### Using `datasets` library
    ```
    from datasets import load_dataset
    dset = datasets.load_dataset("SEACrowd/{dset_name}", trust_remote_code=True)
    ```
    ### Using `seacrowd` library
    ```import seacrowd as sc
    # Load the dataset using the default config
    dset = sc.load_dataset("{dset_name}", schema="seacrowd")
    # Check all available subsets (config names) of the dataset
    print(sc.available_config_names("{dset_name}"))
    # Load the dataset using a specific config
    dset = sc.load_dataset_by_config_name(config_name="<config_name>")
    ```
    
    More details on how to load the `seacrowd` library can be found [here](https://github.com/SEACrowd/seacrowd-datahub?tab=readme-ov-file#how-to-use).
    '''
    readme_string += f'\n\n## Dataset Homepage\n\n[{homepage}]({homepage})'
    readme_string += f'\n\n## Dataset Version\n\nSource: {source_version}. SEACrowd: {seacrowd_version}.'
    readme_string += f'\n\n## Dataset License\n\n{license}'
    readme_string += f'\n\n## Citation\n\nIf you are using the **{dset_name.replace("_", " ").title()}** dataloader in your work, please cite the following:'
    readme_string = re.sub(r"( )+\#", "#", readme_string)
    readme_string = re.sub(r"( )+\`\`\`", "```", readme_string)
    readme_string = re.sub(r"( ){2, 4}", "", readme_string)
    readme_string += f'\n```\n{citation}\n{_SEACROWD_CITATION}\n```'
    readme_string = re.sub(r"( )+\@", "@", readme_string)
    return readme_string


if __name__ == "__main__":
    api = HfApi(
        endpoint="https://huggingface.co",
        token=os.getenv("HF_TOKEN"))
    
    requirements_file = BytesIO(str.encode("seacrowd>=0.2.0"))
    
    # for dirname in ["indolem_sentiment"]:
    for i, dirname in enumerate(os.listdir(_SEA_DATASETS_PATH)):
        if not os.path.isdir(f"{_SEA_DATASETS_PATH}/{dirname}/"):
            print(f"{dirname} is not a directory.")
            continue
        
        print(f'({i} / {len(os.listdir(_SEA_DATASETS_PATH))}) {dirname}')

        api.create_repo(
            f"SEACrowd/{dirname}",
            repo_type="dataset",
            exist_ok=True)

        api.upload_file(
            path_or_fileobj=requirements_file,
            path_in_repo="requirements.txt",
            repo_id=f"SEACrowd/{dirname}",
            repo_type="dataset",
        )
        
        license_file = BytesIO(str.encode(
            import_from(f"seacrowd.sea_datasets.{dirname}.{dirname}", "_LICENSE")))
        api.upload_file(
            path_or_fileobj=license_file,
            path_in_repo="LICENSE",
            repo_id=f"SEACrowd/{dirname}",
            repo_type="dataset",
        )

        readme_file = BytesIO(str.encode(construct_readme(dirname)))
        api.upload_file(
            path_or_fileobj=readme_file,
            path_in_repo="README.md",
            repo_id=f"SEACrowd/{dirname}",
            repo_type="dataset",
        )

        for dataloader_py_file in os.listdir(f"{_SEA_DATASETS_PATH}/{dirname}"):
            if dataloader_py_file.endswith(".py"):
                dataloader_file = f"{_SEA_DATASETS_PATH}/{dirname}/{dataloader_py_file}"
                api.upload_file(
                    path_or_fileobj=dataloader_file,
                    path_in_repo=dataloader_py_file,
                    repo_id=f"SEACrowd/{dirname}",
                    repo_type="dataset",
                )